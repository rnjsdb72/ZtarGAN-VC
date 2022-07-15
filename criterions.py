import torch

def gaussian_pdf(x, mu, sigmasq):
    # adapted from https://mikedusenberry.com/mixture-density-networks
    return (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp((-1/(2*sigmasq)) * ((x-mu)**2))


def loss_fn(mu, sigmasq, pi, target, num_mixtures=10):
    # adapted from https://mikedusenberry.com/mixture-density-networks
    likelihood_z_x = gaussian_pdf(target.unsqueeze(-1), mu, sigmasq) + 1e-5 # add small positive constant
    prior_z = pi+1e-5  # add small positive constant (to avoid nans)
    losses = (prior_z * likelihood_z_x).sum(axis=-1)
    loss = torch.mean(-torch.log(losses))
    return loss