import os
import sys
sys.path.append(os.getcwd())
import jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad
import jax
import torch
import optax
from tqdm import tqdm
import argparse
import equinox as eqx
import time
from functools import partial
from omegaconf import OmegaConf

from inception.hclt import HCLTInception
from data.datasets import load_debd, DEBD, NumpyLoader
from data.image.utils import instantiate_from_config, collect_data_from_dsets


def split_model(model):
    rest, static = eqx.partition(model, eqx.is_array)
    params, traced = eqx.partition(rest, eqx.is_inexact_array)
    
    return params, traced, static

def combine_model(params, traced, static):
    return eqx.combine(eqx.combine(params, traced), static)

@partial(jit, static_argnums=(2,4))
def loss_nll(params, traced, static, mb_data, modarg=False):
    """Negative log likelihood loss function, interpreting the model as a (unnormalized) probability distribution."""
    model = combine_model(params, traced, static)
    fn = (lambda x: model.forward(x).squeeze()[0]) if modarg else lambda x: model.forward(x).squeeze()
    loss = - jnp.sum(
        vmap(fn,
                in_axes = 0,
                out_axes= 0
            )(mb_data) \
    ).squeeze()
    loss += len(mb_data) * model.norm().squeeze()[0] if modarg else len(mb_data) * model.norm().squeeze()
    return loss

@partial(jit, static_argnums=(2, 5, 6, 7))
def mb_step_eqx(params, traced, static, mb_data, opt_state, optimizer, loss_fn, wirtinger=False):
    mb_loss, grads = value_and_grad(loss_fn)(params, traced, static, mb_data)
    if wirtinger:
        grads = jax.tree_util.tree_map(lambda x: jnp.conj(x), grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, mb_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HCLT MNIST')
    parser.add_argument('-w',  '--w_dim',     type=int,   default=4,    help='w dim')
    parser.add_argument('-bs',  '--batch_size',     type=int,   default=250,    help='batch size')
    parser.add_argument('-ep',  '--epochs',    type=int,   default=50, help='number of training steps')
    parser.add_argument('-ds', '--dataset', type=str, default='imagenet32', help='dataset: imagenet32, imagenet64, or one of DEBD datasets')
    parser.add_argument('-u', '--u_dim', type=int, default=4, help='number of u dims')
    parser.add_argument('-dv', '--device', type=int, default=0, help='device')
    parser.add_argument('-p', '--param_type', type=str, default='complex', help='parameter type: positive, complex, negative')
    parser.add_argument('-sd', '--seed', type=int, default=0)
    parser.add_argument('-ns', '--num_samples', type=int, default=20000, help='number of samples to collect from dataset for HCLT construction')

    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    jax.config.update('jax_default_device', jax.devices()[args.device])
    key = random.PRNGKey(args.seed)

    assert args.dataset in ['imagenet32', 'imagenet64'] + DEBD

    if args.dataset in ['imagenet32', 'imagenet64']:
        data_config = OmegaConf.load(f"data/image/{args.dataset}_lossy.yaml")
        if args.batch_size > 0:
            data_config["params"]["batch_size"] = args.batch_size
        dsets = instantiate_from_config(data_config)
        dsets.prepare_data()
        dsets.setup()
        train_loader = dsets._train_dataloader()
        test_loader = dsets._val_dataloader()

        model = HCLTInception(x = collect_data_from_dsets(dsets, num_samples = args.num_samples, split = "train").cuda(),
                              num_W_latents = args.w_dim,
                              num_U_latents = args.u_dim,
                              num_cats = 256,
                              key = key,
                              param_type = args.param_type)
    elif args.dataset in DEBD:
        train_data, test_data, valid_data = load_debd(args.dataset)
        train_data, test_data, valid_data = torch.from_numpy(train_data), torch.from_numpy(test_data), torch.from_numpy(valid_data)

        train_loader = NumpyLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = NumpyLoader(test_data, batch_size=args.batch_size)

        model = HCLTInception(x = train_data.float().to(device),
                                num_W_latents = args.w_dim,
                                num_U_latents = args.u_dim,
                                num_cats = 2,
                                key = key,
                                param_type = args.param_type)

    print("Model created")

    params, traced, static = split_model(model)

    print("Number of parameters: ", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)

    step = 0

    # squared
    loss_fn = loss_nll if args.param_type == 'positive' else partial(loss_nll, modarg=True)


    for epoch in range(args.epochs):
        train_ll = 0
        t0 = time.time()
        tot_samples = 0
        for batch in tqdm(train_loader):
            params, opt_state, mb_loss = mb_step_eqx(params, traced, static, batch, opt_state, 
                                                     optimizer, loss_fn, wirtinger=args.param_type == 'complex')
            train_ll += -mb_loss
            tot_samples += len(batch)

        train_ll /= tot_samples
        

        t1 = time.time()
        test_ll = 0
        tot_samples=0
        for mb_data in tqdm(test_loader):
            test_ll += -loss_fn(params, traced, static, mb_data)
            tot_samples += len(mb_data)
        test_ll /= tot_samples

        t2 = time.time()
        print(f"[Epoch {epoch + 1}/{args.epochs}][train LL: {train_ll:.2f}; val LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; val forward {t2-t1:.2f}] ")