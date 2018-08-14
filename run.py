from openmdao.api import Problem, pyOptSparseDriver, SqliteRecorder
from damage_components import Blade_Damage
import numpy as np
from FAST_util import setupFAST, define_des_var_domains, initialize_rotor_dv, setup_FAST_seq_run_des_var

if __name__ == "__main__":

    FASTinfo = dict()

    # incorporate dynamic response
    FASTinfo['opt_without_FAST'] = False
    FASTinfo['opt_with_FAST_in_loop'] = False
    FASTinfo['calc_fixed_DEMs'] = True
    FASTinfo['calc_fixed_DEMs_seq'] = False
    FASTinfo['opt_with_fixed_DEMs'] = False
    FASTinfo['opt_with_fixed_DEMs_seq'] = False
    FASTinfo['calc_surr_model'] = False
    FASTinfo['opt_with_surr_model'] = False

    description = 'test'

    FASTinfo, blade_damage = setupFAST(FASTinfo, description)


    # === initialize === #
    blade_damage.root = Blade_Damage(FASTinfo=FASTinfo, naero=17, nstr=38)

    if FASTinfo['Use_FAST_sm']:
        FASTinfo, blade_damage = define_des_var_domains(FASTinfo, blade_damage)

    #
    blade_damage.setup(check=False)

    if FASTinfo['seq_run']:

        blade_damage = setup_FAST_seq_run_des_var(blade_damage, FASTinfo)

    elif FASTinfo['use_FAST']:

        if FASTinfo['train_sm']:

            print("Creating Surrogate Model...")

            blade_damage['chord_sub'] = FASTinfo['chord_sub_init']
            blade_damage['r_max_chord'] = 1.0 / (len(blade_damage['chord_sub']) -1.0)
            blade_damage['theta_sub'] = FASTinfo['theta_sub_init']

            blade_damage['sparT'] = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])
            blade_damage['teT'] = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])

        else:

            blade_damage = initialize_rotor_dv(FASTinfo, blade_damage)

    else:

        # not using FAST in the loop, so either using surrogate model or just RotorSE
        blade_damage = initialize_rotor_dv(FASTinfo, blade_damage)




    blade_damage.run()

