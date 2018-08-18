from openmdao.api import Problem, pyOptSparseDriver, SqliteRecorder
from damage_components import Blade_Damage
import numpy as np
from FAST_util import setupFAST, define_des_var_domains, initialize_rotor_dv, setup_FAST_seq_run_des_var
import os

if __name__ == "__main__":

    FASTinfo = dict()

    # incorporate dynamic response
    FASTinfo['calc_fixed_DEMs'] = False
    FASTinfo['calc_fixed_DEMs_seq'] = False
    FASTinfo['calc_surr_model'] = False
    FASTinfo['opt_with_surr_model'] = True  # doesn't do optimization, but rather calculates DEMs and extreme moments based on already trained data

    # unused options for optimization, but still need to set to boolean False
    FASTinfo['opt_without_FAST'] = False
    FASTinfo['opt_with_FAST_in_loop'] = False
    FASTinfo['opt_with_fixed_DEMs'] = False
    FASTinfo['opt_with_fixed_DEMs_seq'] = False

    # only used when doing blade optimization
    FASTinfo['opt_with_fatigue'] = False

    description = 'trained_data'

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

        blade_damage['turbine_class'] = FASTinfo['turbine_class']  # (Enum): IEC turbine class
        blade_damage['turbulence_class'] = FASTinfo['turbulence_class']  # (Enum): IEC turbulence class class

    # === airfoil files ===
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
    airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
    airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
    airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
    airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
    airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

    # place at appropriate radial stations
    blade_damage['af_idx'] = np.array([0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7])

    blade_damage['airfoil_types'] = airfoil_types  # (List): names of airfoil file or initialized CCAirfoils

    # === blade grid ===
    blade_damage['initial_aero_grid'] = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
                                           0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333,
                                           0.88888943, 0.93333333,
                                           0.97777724])  # (Array): initial aerodynamic grid on unit radius
    blade_damage['initial_str_grid'] = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
                                          0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228,
                                          0.06666667, 0.089508406455,
                                          0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333,
                                          0.276686558545, 0.3, 0.333640766319,
                                          0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667,
                                          0.602196371696, 0.63333333,
                                          0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333,
                                          0.88888943, 0.93333333, 0.97777724,
                                          1.0])  # (Array): initial structural grid on unit radius

    blade_damage['rstar_damage'] = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments

    FASTinfo['nBlades'] = 3
    blade_damage['nBlades'] = 3
    blade_damage['bladeLength'] = FASTinfo['bladeLength']

    blade_damage.run()

