

class SelectChannel():
    def __init__(self, channel, channel_last = True):
        self.channel = channel
        self.channel_last = channel_last

    def __call__(self, x):
        if self.channel_last:
            return x[:,:,self.channel]
        else:
            return x[self.channel,:,:]