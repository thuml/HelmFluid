from models import HelmFluid_2D_corr_potential, HelmFluid_2D_corr_potential_boundary, HelmFluid_2D_corr_potential_128, \
    HelmFluid_2D_corr_potential_256, HelmFluid_2D_corr_potential_boundary_128, HelmFluid_2D_corr_potential_sea, HelmFluid_3D_boundary_32



def get_model(args):
    model_dict = {
        'HelmFluid_2D_corr_potential': HelmFluid_2D_corr_potential,
        'HelmFluid_2D_corr_potential_sea': HelmFluid_2D_corr_potential_sea,
        'HelmFluid_2D_corr_potential_128': HelmFluid_2D_corr_potential_128,
        'HelmFluid_2D_corr_potential_256': HelmFluid_2D_corr_potential_256,
        'HelmFluid_2D_corr_potential_boundary': HelmFluid_2D_corr_potential_boundary,
        'HelmFluid_2D_corr_potential_boundary_128': HelmFluid_2D_corr_potential_boundary_128,
        'HelmFluid_3D_boundary_32': HelmFluid_3D_boundary_32,
    }
    return model_dict[args.model].Model(args).cuda()
