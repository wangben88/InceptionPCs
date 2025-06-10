import torch


class RGB2YCoCg():
    def __init__(self, channel_last = False, rgb_range = (-1, 1)):
        self.channel_last = channel_last
        self.rgb_range = rgb_range

    def __call__(self, x):
        if self.channel_last:
            R, G, B = x[:,:,0], x[:,:,1], x[:,:,2]
        else:
            R, G, B = x[0,:,:], x[1,:,:], x[2,:,:]

        R = (R - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0])
        G = (G - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0])
        B = (B - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0])

        Co  = R - B;
        tmp = B + Co/2;
        Cg  = G - tmp;
        Y   = tmp + Cg/2;

        # Make the range of Y to be [-1, 1]
        Y = Y * 2 - 1

        if self.channel_last:
            return torch.stack((Y, Co, Cg), dim = 2)
        else:
            return torch.stack((Y, Co, Cg), dim = 0)


class RGB2YCoCgR():
    def __init__(self, channel_last = False, rgb_range = (-1, 1)):
        self.channel_last = channel_last
        self.rgb_range = rgb_range

    def __call__(self, x):
        if self.channel_last:
            R, G, B = x[:,:,0], x[:,:,1], x[:,:,2]
        else:
            R, G, B = x[0,:,:], x[1,:,:], x[2,:,:]

        R = ((R - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0]) * 255).long()
        G = ((G - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0]) * 255).long()
        B = ((B - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0]) * 255).long()

        Co  = R - B
        tmp = B + Co // 2
        Cg  = G - tmp
        Y   = tmp + Cg // 2

        # #### Verification
        # tmp = Y - Cg//2
        # G2   = Cg + tmp
        # B2   = tmp - Co//2
        # R2   = B2 + Co

        # assert (R == R2).all() and (G == G2).all() and (B == B2).all()
        # ####

        Co += 256
        Cg += 256

        # assert Y.min() >= 0 and Y.max() < 256
        # assert Co.min() >= 0 and Co.max() < 512
        # assert Cg.min() >= 0 and Cg.max() < 512

        if self.channel_last:
            return torch.stack((Y, Co, Cg), dim = 2)
        else:
            return torch.stack((Y, Co, Cg), dim = 0)


class YCoCg2RGB():
    def __init__(self, channel_last = True):
        self.channel_last = channel_last

    def __call__(self, x):

        assert x.min() >= -1 and x.max() <= 1

        if self.channel_last:
            Y, Co, Cg = x[:,:,0], x[:,:,1], x[:,:,2]
        else:
            Y, Co, Cg = x[0,:,:], x[1,:,:], x[2,:,:]

        # Convert the range of Y back to [0, 1]
        Y = (Y + 1) / 2

        tmp = Y - Cg/2;
        G   = Cg + tmp;
        B   = tmp - Co/2;
        R   = B + Co;

        if self.channel_last:
            return torch.stack((R, G, B), dim = 2)
        else:
            return torch.stack((R, G, B), dim = 0)