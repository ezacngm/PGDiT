import torch
import numpy as np
from scipy.special import gammaln
from mask_generator import MaskGenerator_progressive
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf

class SphericalHarmonicProcessor_th:
    '''Spherical Harmonic Baseline Processor'''

    def __init__(self, sh_order=10, smooth=0.0001,mask=None,mask_bvecs=None,mixed=False):
        self.sh_order = sh_order
        self.smooth = smooth
        self.mask = mask
        B,T,W,H = mask.shape
        self.bvec_mask = mask_bvecs
        mask_generator = MaskGenerator_progressive((T,W,H))
        self.mask, self.bvec_mask = mask_generator(B, "cuda")
    def run_subject(self, dmri, bvec,mask = None,bvec_mask=None):

        bvec = bvec.squeeze()
        self.mask = self.mask.squeeze()
        T,W,H = self.mask.shape
        self.bvec_mask = self.bvec_mask.squeeze(0)#
        dmri = dmri.squeeze()
        # mask_generator = MaskGenerator_progressive((T,W,H))
        # self.mask, self.bvec_mask = mask_generator(B, "cuda")
        bvecs_in = bvec * (1 - self.bvec_mask)
        bvecs_in = bvecs_in
        nonzero_mask = ~(bvecs_in == 0).all(dim=1)
        bvecs_in = bvecs_in[nonzero_mask, :]
        bvecs_out = bvec
        dmri_in = dmri * (1 - self.mask)
        # nonzero_mask = ~(dmri_in == 0).all(dim=(1, 2))
        dmri_in_filtered = dmri_in[nonzero_mask,:, :]
        dmri_in_filtered = dmri_in_filtered.unsqueeze(1)
        assert dmri_in_filtered.shape[0]==bvecs_in.shape[0]

        sh_coeffs = get_spherical_harmonics_coefficients(
            dmri_in_filtered, bvecs_in,device=dmri.device, sh_order=self.sh_order, smooth=self.smooth
        )
        recon = sh_to_sf(sh_coeffs, bvecs_out, sh_order=self.sh_order, device=dmri.device)
        return sh_coeffs, recon.permute(3,0,1,2)
    def get_sh_coeffs(self, dmri,bvecs):

        sh_coeffs = get_spherical_harmonics_coefficients(dmri, bvecs,device=dmri.device, sh_order=self.sh_order, smooth=self.smooth)
        return sh_coeffs
def get_spherical_harmonics_coefficients(dwi, bvecs, device, sh_order=8, smooth=0.006):

    theta,phi = HemiSphere_torch(xyz=bvecs)
    Ba, m, n = sph_harm_basis_torch(sh_order, theta, phi, device)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L.float())
    data_sh = torch.matmul(dwi.permute(1,2,3,0), invB.t())
    return data_sh


def smooth_pinv(B, L):
    L = torch.diag(L)
    inv = torch.pinverse(torch.cat((B, L)))
    return inv[:, :len(B)]


def sph_harm_basis_torch(sh_order, theta, phi,device):

    m, n = sph_harm_ind_list_torch(sh_order, device)
    phi = torch.reshape(phi, (-1, 1))
    theta = torch.reshape(theta, (-1, 1))

    m = -m
    real_sh = real_sph_harm_torch(m, n, theta, phi)
    # real_sh /= np.where(m == 0, 1., np.sqrt(2))
    return real_sh, m, n


def real_sph_harm_torch(m, n, theta, phi):

    sh = spherical_harmonics_torch(torch.abs(m), n, phi, theta)

    real_sh = torch.where(m > 0, sh[:,:,1], sh[:,:,0])
    # real_sh = real_sh * torch.where(m == 0, torch.tensor(1.,device=phi.device), torch.tensor(np.sqrt(2),device=phi.device))
    return real_sh


def spherical_harmonics_torch(m, n, theta, phi):
    x = torch.cos(phi)
    val = legendre_associated(m, n, x)
    val = val * torch.sqrt((2 * n.float() + 1) / 4.0 / np.pi)
    val = val * torch.tensor(np.exp(0.5 * (gammaln(n.cpu().numpy() - m.cpu().numpy() + 1) - gammaln(n.cpu().numpy() + m.cpu().numpy() + 1))),device=phi.device).float()
    val = val.unsqueeze(-1) * torch.stack([torch.cos(m.float() * theta),torch.sin(m.float() * theta)],dim=-1)
    return val


def legendre_associated(m, n, x):
    x=x.squeeze(1)
    ans=torch.zeros(x.shape[0],m.shape[0],device=x.device)
    somx2 = torch.sqrt(1.0 - x * x+1e-8)
    for j in range(m.shape[0]):
        cx_list=[torch.ones(x.shape[0],device=x.device)]
        fact = 1.0
        for i in range(0, m[j]):
            # cx[:,m[j]] = - cx[:,m[j]] * fact * somx2
            cx_list[0] = -cx_list[0] * fact * somx2
            fact = fact + 2.0

        # cx_list = [cx[:,m[j]]]
        if (m[j] != n[j]):
            cx_list.append(x * float(2 * m[j] + 1) * cx_list[0])
            # cx[:,m[j] + 1] = x * float(2 * m[j] + 1) * cx[:,m[j]]

            for i in range(m[j] + 2, n[j] + 1):
                cx_list.append((float(2 * i - 1) * x * cx_list[i - 1-m[j]] + float(- i - m[j] + 1) * cx_list[i - 2-m[j]]) / float(i - m[j]))
                # cx[:,i] = (float(2 * i - 1) * x * cx[:,i - 1] + float(- i - m[j] + 1) * cx[:,i - 2]) / float(i - m[j])
        ans[:,j]=cx_list[-1]
    return ans


def sph_harm_ind_list_torch(sh_order,device):

    if sh_order % 2 != 0:
        raise ValueError('sh_order must be an even integer >= 0')

    n_range = torch.arange(0, sh_order + 1, 2, device=device)
    n_list = n_range.repeat_interleave(n_range * 2 + 1)

    ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
    offset = 0
    m_list = torch.empty(ncoef, dtype=n_list.dtype,device=device)
    for ii in n_range:
        m_list[offset:offset + 2 * ii + 1] = torch.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return (m_list, n_list)

def HemiSphere_torch(xyz):
    xyz = xyz * (1 - 2 * torch.lt(xyz[:, -1:],0).float()) # to remove if we can assume xyz on HemSphere
    theta, phi = cart2sphere(xyz)
    return theta, phi


def cart2sphere(xyz):

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = torch.sqrt(x * x + y * y + z * z)
    theta = torch.acos(z/r)
    # theta = torch.where(r > 0, theta, 0.)
    phi = torch.atan2(y, x)
    return theta, phi


def sh_to_sf(sh_coefficients, bvecs, sh_order, device):

    theta, phi = HemiSphere_torch(xyz=bvecs)
    Ba, m, n = sph_harm_basis_torch(sh_order, theta, phi, device)
    data_resampled = torch.matmul(sh_coefficients, Ba.t())
    return data_resampled