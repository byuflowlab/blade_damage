from openmdao.api import IndepVarComp, Component, Group, ParallelFDGroup
import numpy as np
import sys
import os
from distutils.dir_util import copy_tree
from akima import Akima, akima_interp_with_derivs
import matlab.engine
import re
from scipy import stats
import os
import time
import matplotlib.pyplot as plt
import pickle
from enum import Enum

# AeroelasticSE
sys.path.insert(0, '../RotorSE_FAST/AeroelasticSE/src/AeroelasticSE/FAST_mdao')

# rainflow
sys.path.insert(0, '../RotorSE_FAST/AeroelasticSE/src/AeroelasticSE/rainflow')

# ===================== OpenMDAO Components and Groups ===================== #

class CreateFASTConfig(Component):
    def __init__(self, naero, nstr, FASTinfo, WNDfile_List, caseids):
        super(CreateFASTConfig, self).__init__()

        self.caseids = caseids
        self.WNDfile_List = WNDfile_List

        self.dT = FASTinfo['dT']
        self.description = FASTinfo['description']
        self.path = FASTinfo['path']
        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd_config']
        self.sgp = FASTinfo['sgp']

        self.nonturb_dir = FASTinfo['nonturb_wnd_dir']
        self.turb_dir = FASTinfo['turb_wnd_dir']
        self.wndfiletype = FASTinfo['wnd_type_list']
        self.parked_type = FASTinfo['parked']

        self.Tmax_turb = FASTinfo['Tmax_turb']
        self.Tmax_nonturb = FASTinfo['Tmax_nonturb']
        self.rm_time = FASTinfo['rm_time']

        self.FAST_opt_directory = FASTinfo['opt_dir']
        self.template_dir = FASTinfo['template_dir']
        self.fst_exe = FASTinfo['fst_exe']

        self.train_sm = FASTinfo['train_sm']
        if self.train_sm:
            self.sm_dir = FASTinfo['sm_dir']

        self.check_stif_spline = FASTinfo['check_stif_spline']

        self.output_list = FASTinfo['output_list']

        # used to train surrogate model using WindPact turbine designs
        self.run_template_files = FASTinfo['run_template_files']
        self.set_chord_twist = FASTinfo['set_chord_twist']
        self.set_blade_length = FASTinfo['bladeLength']

        self.FAST_template_name = FASTinfo['FAST_template_name']

        # add necessary parameters
        self.add_param('nBlades', val=0)

        self.add_param('EIyy', val=np.zeros(nstr))

        self.add_param('r_max_chord', val=0.0)
        self.add_param('chord_sub', val=np.zeros(4))
        self.add_param('theta_sub', val=np.zeros(4))
        self.add_param('idx_cylinder_aero', val=0)
        self.add_param('initial_aero_grid', val=np.zeros(naero))

        self.add_param('rho', val=0.0)
        self.add_param('control:tsr', val=0.0)
        self.add_param('g', val=0.0)
        self.add_param('hubHt', np.zeros(1))
        self.add_param('mu', val=0.0)
        self.add_param('precone', val=0.0)
        self.add_param('tilt', val=0.0)
        self.add_param('hubFraction', val=0.0)
        self.add_param('leLoc', val=np.zeros(nstr))

        self.add_param('FAST_Chord_Aero', val=np.zeros(naero))
        self.add_param('FAST_Theta_Aero', val=np.zeros(naero))

        self.add_param('FAST_Chord_Str', val=np.zeros(nstr))
        self.add_param('FAST_Theta_Str', val=np.zeros(nstr))

        self.add_param('FAST_r_Aero', val=np.zeros(naero))
        self.add_param('FAST_precurve_Aero', val=np.zeros(naero))
        self.add_param('FAST_precurve_Str', val=np.zeros(nstr))
        self.add_param('FAST_Rhub', val=0.0)
        self.add_param('FAST_Rtip', val=0.0)
        self.add_param('V', val=np.zeros(200))

        self.add_param('FlpStff', val=np.zeros(nstr))
        self.add_param('EdgStff', val=np.zeros(nstr))
        self.add_param('GJStff', val=np.zeros(nstr))
        self.add_param('EAStff', val=np.zeros(nstr))
        self.add_param('BMassDen', val=np.zeros(nstr))

        self.add_param('af_idx', val=np.zeros(naero))
        self.add_param('airfoil_types', val=np.zeros(8))

        # Set all constraints to be calculated using finite difference method
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1.0e-6

        # add_output
        self.add_output('cfg_master', val=dict(),pass_by_obj=False)

    def solve_nonlinear(self, params, unknowns, resids):

        # create file directory for each surrogate model training point
        if self.train_sm:
            FAST_opt_directory = self.sm_dir
        else:
            FAST_opt_directory = self.FAST_opt_directory

        # needs to be created just once for optimization
        if os.path.isdir(FAST_opt_directory):
            pass
        else:
            os.mkdir(FAST_opt_directory)

        # === create config === #

        # Setup input config dictionary of dictionaries.
        caseids = self.caseids
        cfg_master = {}  # master config dictionary (dictionary of dictionaries)

        for sgp in range(0,len(self.sgp)):

            sgp_dir = FAST_opt_directory + '/' + 'sgp' + str(self.sgp[sgp])

            for wnd_file in range(0, len(self.WNDfile_List)):

                # determine FAST_wnd_directory
                spec_caseid = sgp*len(self.WNDfile_List) + wnd_file
                FAST_sgp_directory = sgp_dir
                FAST_wnd_directory = sgp_dir + '/' + caseids[spec_caseid]

                # needs to be created for each .wnd input file
                if not os.path.isdir(FAST_sgp_directory):
                    os.mkdir(FAST_sgp_directory)

                if os.path.isdir(FAST_wnd_directory):
                    pass
                else:
                    os.mkdir(FAST_wnd_directory)
                    copy_tree(self.template_dir, FAST_wnd_directory)

                # Create dictionary for this particular index
                cfg = {}

                # === run files/directories === #
                cfg['fst_masterfile'] = self.FAST_template_name + '.fst'

                cfg['fst_runfile'] = 'fst_runfile.fst'

                cfg['fst_masterdir'] = FAST_wnd_directory

                cfg['fst_rundir'] = cfg['fst_masterdir']

                cfg['fst_exe'] = self.fst_exe

                cfg['fst_file_type'] = 0
                cfg['ad_file_type'] = 1

                def replace_line(file_name, line_num, text):
                    lines = open(file_name, 'r').readlines()
                    lines[line_num] = text
                    out = open(file_name, 'w')
                    out.writelines(lines)
                    out.close()

                # exposed parameters (no corresponding RotorSE parameter)
                if self.wndfiletype[spec_caseid] == 'turb':
                    cfg['TMax'] = self.Tmax_turb
                else:
                    cfg['TMax'] = self.Tmax_nonturb
                cfg['DT'] = self.dT
                cfg['TStart'] = self.rm_time

                # === Add .wnd file location to Aerodyn.ipt file === #

                if self.wndfiletype[spec_caseid] == 'turb':
                    wnd_file_path = self.path + self.turb_dir + self.WNDfile_List[wnd_file]
                else:
                    wnd_file_path = self.path + self.nonturb_dir + self.WNDfile_List[wnd_file]

                aerodyn_file_name = cfg['fst_masterdir'] + '/' + self.FAST_template_name + '_AD.ipt'
                replace_line(aerodyn_file_name, 9, wnd_file_path + '\n')

                # === parked configuration === #
                if self.parked_type[wnd_file] == 'yes':
                    cfg['TimGenOn'] = 9999.9

                cfg['OutFileFmt'] = 3  # text and binary output files

                # # exposed parameters (no corresponding RotorSE parameter)
                cfg['SysUnits'] = 'SI'
                cfg['StallMod'] = 'BEDDOES'
                cfg['UseCm'] = 'NO_CM'
                cfg['InfModel'] = 'DYNIN'
                cfg['AToler'] = 0.005
                cfg['TLModel'] = 'PRANDtl'
                cfg['HLModel'] = 'NONE'
                cfg['TwrShad'] = 0.0
                cfg['ShadHWid'] = 9999.9
                cfg['T_Shad_Refpt'] = 9999.9
                cfg['DTAero'] = 0.02479

                # strain gage placement for bending moment
                cfg['NBlGages'] = self.NBlGages[sgp]
                cfg['BldGagNd'] = self.BldGagNd[sgp]

                if self.set_chord_twist:

                    f = open(aerodyn_file_name, "r")

                    lines = f.readlines()

                    if self.FAST_template_name == 'NREL5MW':
                        lines = lines[28:45]
                    else:
                        lines = lines[24:41]

                    DR_nodes = []

                    for i in range(len(lines)):
                        lines[i] = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lines[i].strip('\n'))
                        DR_nodes.append(float(lines[i][0]))

                    chord_spline = Akima(np.linspace( 0, 1, len(params['chord_sub']) ), params['chord_sub'])
                    twist_spline = Akima(np.linspace( 0, 1, len(params['theta_sub']) ), params['theta_sub'])

                    cfg['Chord'] = chord_spline.interp(np.array(DR_nodes)/self.set_blade_length)[0]
                    cfg['AeroTwst'] = twist_spline.interp(np.array(DR_nodes)/self.set_blade_length)[0]

                elif not self.run_template_files:

                    # === general parameters === #
                    cfg['NumBl'] = params['nBlades']

                    if hasattr(params['g'], "__len__"):
                        cfg['Gravity'] = params['g'][0]
                    else:
                        cfg['Gravity'] = params['g']

                    cfg['RotSpeed'] = params['control:tsr']
                    cfg['TipRad'] = params['FAST_Rtip']
                    cfg['HubRad'] = params['FAST_Rhub']
                    cfg['ShftTilt'] = params['tilt']
                    cfg['PreCone1'] = params['precone']
                    cfg['PreCone2'] = params['precone']
                    cfg['PreCone3'] = params['precone']

                    # # Aerodyn File

                    # Add DLC .wnd file name to Aerodyn.ipt input file
                    cfg['HH'] = params['hubHt'][0]

                    if hasattr(params['rho'], "__len__"):
                        cfg['AirDens'] = params['rho'][0]
                        cfg['KinVisc'] = params['mu'][0] / params['rho'][0]
                    else:
                        cfg['AirDens'] = params['rho']
                        cfg['KinVisc'] = params['mu'] / params['rho']

                    # cfg['FoilNm'] = FoilNm
                    cfg['NFoil'] = (params['af_idx'] + np.ones(np.size(params['af_idx']))).astype(int)

                    cfg['BldNodes'] = np.size(params['af_idx'])

                    # Make akima splines of RNodes/AeroTwst and RNodes/Chord
                    theta_sub_spline = Akima(params['FAST_r_Aero'], params['FAST_Theta_Aero'])
                    chord_sub_spline = Akima(params['FAST_r_Aero'], params['FAST_Chord_Aero'])

                    # Redefine RNodes so that DRNodes can be calculated using AeroSubs
                    RNodes = params['FAST_r_Aero']
                    RNodes = np.linspace(RNodes[0], RNodes[-1], len(RNodes))

                    cfg['RNodes'] = RNodes
                    # Find new values of AeroTwst and Chord using redefined RNodes

                    FAST_Theta = theta_sub_spline.interp(RNodes)[0]
                    FAST_Chord = chord_sub_spline.interp(RNodes)[0]

                    cfg['Chord'] = FAST_Chord
                    cfg['AeroTwst'] = FAST_Theta

                    DRNodes = np.zeros(np.size(params['af_idx']))
                    for i in range(0, np.size(params['af_idx'])):
                        if i == 0:
                            DRNodes[i] = 2.0 * (RNodes[0] - params['FAST_Rhub'])
                        else:
                            DRNodes[i] = 2.0 * (RNodes[i] - RNodes[i - 1]) - DRNodes[i - 1]

                    cfg['DRNodes'] = DRNodes

                    # # Blade File

                    cfg['NBlInpSt'] = len(params['FlpStff'])
                    cfg['BlFract'] = np.linspace(0, 1, len(params['FlpStff']))
                    cfg['AeroCent'] = params['leLoc']
                    cfg['StrcTwst'] = params['FAST_Theta_Str']
                    cfg['BMassDen'] = params['BMassDen']

                    cfg['FlpStff'] = params['FlpStff']
                    cfg['EdgStff'] = params['EdgStff']
                    cfg['GJStff'] = params['GJStff']
                    cfg['EAStff'] = params['EAStff']

                    # exposed parameters (no corresponding RotorSE parameter)
                    cfg['CalcBMode'] = 'False'
                    cfg['BldFlDmp1'] = 2.477465
                    cfg['BldFlDmp2'] = 2.477465
                    cfg['BldEdDmp1'] = 2.477465
                    cfg['FlStTunr1'] = 1.0
                    cfg['FlStTunr2'] = 1.0
                    cfg['AdjBlMs'] = 1.04536
                    cfg['AdjFlSt'] = 1.0
                    cfg['AdjEdSt'] = 1.0

                    # unused parameters (not used by FAST)
                    alpha = 0.5 * np.arctan2(2 * params['EAStff'], params['FlpStff'] - params['EAStff'])
                    for i in range(0, len(alpha)):
                        alpha[i] = min(0.99999, alpha[i])
                    cfg['Alpha'] = alpha

                    cfg['PrecrvRef'] = np.zeros(len(params['FlpStff']))
                    cfg['PreswpRef'] = np.zeros(len(params['FlpStff']))
                    cfg['FlpcgOf'] = np.zeros(len(params['FlpStff']))
                    cfg['Edgcgof'] = np.zeros(len(params['FlpStff']))
                    cfg['FlpEAOf'] = np.zeros(len(params['FlpStff']))
                    cfg['EdgEAOf'] = np.zeros(len(params['FlpStff']))

                    # compare EI properties
                    BladeStructureProperties = np.loadtxt('FAST_Files/RotorSE_InputFiles/BladeStructureProperties.txt')

                    # Blade Structural Properties
                    #0 BlFract
                    #1 AeroCent
                    #2 StrcTwst
                    #3 BMassDen
                    #4 FlpStff
                    #5 EdgStff
                    #6 GJStff
                    #7 EAStff
                    #8 Alpha
                    #9 FlpIner
                    #10 EdgIner
                    #11 PrecrvRef
                    #12 PreswpRef
                    #13 FlpcgOf
                    #14 EdgcgOf
                    #15 FlpEAOf
                    #16 EdgEAOf

                    # FlpStff, EdgStff, GJStff, EAStff
                    EI_flp_spline = Akima(params['FAST_precurve_Str'], params['FlpStff'])
                    EI_flp = EI_flp_spline.interp(BladeStructureProperties[:, 0])[0]

                    EI_edge_spline = Akima(params['FAST_precurve_Str'], params['EdgStff'])
                    EI_edge = EI_edge_spline.interp(BladeStructureProperties[:, 0])[0]

                    EI_gj_spline = Akima(params['FAST_precurve_Str'], params['GJStff'])
                    EI_gj = EI_gj_spline.interp(BladeStructureProperties[:, 0])[0]

                    EI_ea_spline = Akima(params['FAST_precurve_Str'], params['EAStff'])
                    EI_ea = EI_ea_spline.interp(BladeStructureProperties[:, 0])[0]

                    if self.check_stif_spline:

                        # plots
                        BlFract = BladeStructureProperties[:, 0]
                        FlpStff = BladeStructureProperties[:, 4]
                        EdgStff = BladeStructureProperties[:, 5]
                        GJStff = BladeStructureProperties[:, 6]
                        EAStff = BladeStructureProperties[:, 7]

                        plt.figure()
                        plt.plot(BlFract, EI_flp, label='RotorSE spline')
                        plt.plot(BlFract, FlpStff, label='FAST nominal value')
                        plt.legend()
                        plt.title('FlpStff')

                        plt.figure()
                        plt.plot(BlFract, EI_edge, label='RotorSE spline')
                        plt.plot(BlFract, EdgStff, label='FAST nominal value')
                        plt.legend()
                        plt.title('EdgStff')

                        plt.figure()
                        plt.plot(BlFract, EI_gj, label='RotorSE spline')
                        plt.plot(BlFract, GJStff, label='FAST nominal value')
                        plt.legend()
                        plt.title('GJStff Stiffness')

                        plt.figure()
                        plt.plot(BlFract, EI_ea, label='RotorSE spline')
                        plt.plot(BlFract, EAStff, label='FAST nominal value')
                        plt.legend()
                        plt.title('EAStff Stiffness')

                        plt.show()

                        quit()

                cfg_master[self.caseids[spec_caseid]] = cfg

        unknowns['cfg_master'] = cfg_master


class use_FAST_surr_model(Component):
    def __init__(self, FASTinfo, naero, nstr):
        super(use_FAST_surr_model, self).__init__()

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

        self.FASTinfo = FASTinfo

        self.add_param('r_max_chord', val=0.0)
        self.add_param('chord_sub', val=np.zeros(4))
        self.add_param('theta_sub', val=np.zeros(4))
        self.add_param('sparT', val=np.zeros(5))
        self.add_param('teT', val=np.zeros(5))

        self.approximation_model = FASTinfo['approximation_model']

        self.training_point_dist = FASTinfo['training_point_dist']  # 'linear', 'lhs'

        if self.training_point_dist == 'lhs':
            self.num_pts = FASTinfo['num_pts']

            self.sm_var_file = FASTinfo['sm_var_file_master']
            self.sm_DEM_file = FASTinfo['sm_DEM_file_master']
            self.sm_load_file = FASTinfo['sm_load_file_master']
            self.sm_def_file = FASTinfo['sm_def_file_master']

        else:
            self.sm_var_max = FASTinfo['sm_var_max']

            self.sm_var_file = FASTinfo['sm_var_file']
            self.sm_DEM_file = FASTinfo['sm_DEM_file']

        self.opt_dir = FASTinfo['opt_dir']

        self.var_filename = self.opt_dir + '/' + self.sm_var_file
        self.DEM_filename = self.opt_dir + '/' + self.sm_DEM_file
        self.load_filename = self.opt_dir + '/' + self.sm_load_file
        self.def_filename = self.opt_dir + '/' + self.sm_def_file

        self.dir_saved_plots = FASTinfo['dir_saved_plots']

        self.sm_var_index = FASTinfo['sm_var_index']
        self.var_index = FASTinfo['var_index']
        self.sm_var_names = FASTinfo['sm_var_names']

        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd']

        self.add_param('DEMx', shape=18, desc='DEMx')
        self.add_param('DEMy', shape=18, desc='DEMy')

        self.check_fit = FASTinfo['check_fit']

        self.do_cv_DEM = FASTinfo['do_cv_DEM']
        self.do_cv_Load = FASTinfo['do_cv_Load']
        self.do_cv_def = FASTinfo['do_cv_def']

        self.print_sm = FASTinfo['print_sm']

        if self.do_cv_DEM or self.do_cv_Load or self.do_cv_def:
            self.kfolds = FASTinfo['kfolds']
            self.num_folds = FASTinfo['num_folds']

            self.theta0_val = FASTinfo['theta0_val']

        self.add_output('DEMx_sm', val=np.zeros(18))  # , pass_by_obj=False)
        self.add_output('DEMy_sm', val=np.zeros(18))  # , pass_by_obj=False)

        self.add_output('Edg_sm', val=np.zeros(nstr))  # , pass_by_obj=False)
        self.add_output('Flp_sm', val=np.zeros(nstr))  # , pass_by_obj=False)

        self.add_output('def_sm', val=0.0)  # , pass_by_obj=False)

        self.nstr = nstr

        # to calculate chord / blade length
        self.add_param('bladeLength', shape=1, desc='Blade length')
        self.add_param('nBlades', val=3, desc='Number of Blades')

        # other surrogate model inputs
        self.add_param('turbulence_class', val=Enum('A', 'B', 'C'), desc='IEC turbulence class', pass_by_obj=True)
        self.add_param('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True)
        self.add_param('af_idx', val=np.zeros(naero))
        self.add_param('airfoil_types', val=np.zeros(8))

    def solve_nonlinear(self, params, unknowns, resids):

        # === load surrogate model fits === #
        sm_name_list = ['sm_x', 'sm_y', 'sm_x_load', 'sm_y_load']

        for i in range(len(sm_name_list)):
            pkl_file_name = self.opt_dir + '/' + sm_name_list[i] + '_' + self.approximation_model + '.pkl'

            file_handle = open(pkl_file_name, "r")

            if sm_name_list[i] == 'sm_x':
                sm_x = pickle.load(file_handle)
            elif sm_name_list[i] == 'sm_y':
                sm_y = pickle.load(file_handle)
            elif sm_name_list[i] == 'sm_x_load':
                sm_x_load = pickle.load(file_handle)
            elif sm_name_list[i] == 'sm_y_load':
                sm_y_load = pickle.load(file_handle)
            elif sm_name_list[i] == 'sm_def':
                sm_def = pickle.load(file_handle)

        # === estimate outputs === #

        # current design variable values
        sv = []
        for i in range(0, len(self.sm_var_names)):

            # chord_sub, theta_sub
            if hasattr(params[self.sm_var_names[i]], '__len__'):
                for j in range(0, len(self.sm_var_names[i])):

                    if j in self.sm_var_index[i]:

                        # calculate chord / blade length
                        if self.sm_var_names[i] == 'chord_sub':
                            sv.append( params[self.sm_var_names[i]][j] / ( params['bladeLength'] ) )
                        else:
                            sv.append(params[self.sm_var_names[i]][j])
            # turbulence intensity
            else:
                sv.append(params[self.sm_var_names[i]])

        # === predict values === #

        int_sv = np.zeros([len(sv), 1])
        for i in range(0, len(int_sv)):
            int_sv[i] = sv[i]

        # DEMs
        DEMx_sm = np.transpose(sm_x.predict_values(np.transpose(int_sv)))
        DEMy_sm = np.transpose(sm_y.predict_values(np.transpose(int_sv)))

        # extreme loads
        Edg_sm = np.transpose(sm_x_load.predict_values(np.transpose(int_sv)))
        Flp_sm = np.transpose(sm_y_load.predict_values(np.transpose(int_sv)))

        # tip deflections
        # def_sm = np.transpose(sm_def.predict_values(np.transpose(int_sv)))

        # === === #

        # def_sm = def_sm[0][0]

        unknowns['DEMx_sm'] = DEMx_sm
        unknowns['DEMy_sm'] = DEMy_sm
        unknowns['Edg_sm'] = Edg_sm
        unknowns['Flp_sm'] = Flp_sm

        # unknowns['def_sm'] = def_sm


class calc_FAST_sm_fit(Component):

    def __init__(self, FASTinfo, naero, nstr):
        super(calc_FAST_sm_fit, self).__init__()

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

        self.add_param('r_max_chord', val=0.0)
        self.add_param('chord_sub', val=np.zeros(4))
        self.add_param('theta_sub', val=np.zeros(4))
        self.add_param('sparT', val=np.zeros(5))
        self.add_param('teT', val=np.zeros(5))

        self.FASTinfo = FASTinfo

        self.approximation_model = FASTinfo['approximation_model']

        self.training_point_dist = FASTinfo['training_point_dist'] # 'linear', 'lhs'

        self.calc_DEM_using_sm_no_opt = FASTinfo['calc_DEM_using_sm_no_opt']

        if self.training_point_dist == 'lhs':
            self.num_pts = FASTinfo['num_pts']

            self.sm_var_file = FASTinfo['sm_var_file_master']
            self.sm_DEM_file = FASTinfo['sm_DEM_file_master']
            self.sm_load_file = FASTinfo['sm_load_file_master']
            self.sm_def_file = FASTinfo['sm_def_file_master']

        else:
            self.sm_var_max = FASTinfo['sm_var_max']

            self.sm_var_file = FASTinfo['sm_var_file']
            self.sm_DEM_file = FASTinfo['sm_DEM_file']

        self.opt_dir = FASTinfo['opt_dir']

        self.var_filename = self.opt_dir + '/' + self.sm_var_file
        self.DEM_filename = self.opt_dir + '/' + self.sm_DEM_file
        self.load_filename = self.opt_dir + '/' + self.sm_load_file
        self.def_filename = self.opt_dir + '/' + self.sm_def_file

        self.dir_saved_plots = FASTinfo['dir_saved_plots']

        self.sm_var_index = FASTinfo['sm_var_index']
        self.var_index = FASTinfo['var_index']
        self.sm_var_names = FASTinfo['sm_var_names']

        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd']

        self.add_param('DEMx', shape=18, desc='DEMx')
        self.add_param('DEMy', shape=18, desc='DEMy')

        self.add_param('bladeLength', shape=1, desc='Blade length')
        self.add_param('nBlades', val=3, desc='Number of Blades')


        self.check_fit = FASTinfo['check_fit']

        self.do_cv_DEM = FASTinfo['do_cv_DEM']
        self.do_cv_Load = FASTinfo['do_cv_Load']
        self.do_cv_def = FASTinfo['do_cv_def']

        self.turb_class = FASTinfo['turbulence_class']

        self.check_sm_accuracy = FASTinfo['check_sm_accuracy']

        self.print_sm = FASTinfo['print_sm']

        if self.do_cv_DEM or self.do_cv_Load or self.do_cv_def:

            self.kfolds = FASTinfo['kfolds']
            self.num_folds = FASTinfo['num_folds']

        self.theta0_val = FASTinfo['theta0_val']

        self.nstr = nstr

        # other surrogate model inputs
        self.add_param('turbulence_class', val=Enum('A', 'B', 'C'), desc='IEC turbulence class', pass_by_obj=True)
        self.add_param('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True)
        self.add_param('turbulence_intensity', val=0.14, desc='IEC turbine class intensity', pass_by_obj=True)
        self.add_param('af_idx', val=np.zeros(naero))
        self.add_param('airfoil_types', val=np.zeros(8))

    def solve_nonlinear(self, params, unknowns, resids):

        # get turbulence intensity value
        if params['turbulence_class'] == 'A':
            params['turbulence_intensity'] = 0.12
        elif params['turbulence_class'] == 'B':
            params['turbulence_intensity'] = 0.14
        elif params['turbulence_class'] == 'C':
            params['turbulence_intensity'] = 0.16

        # get rated torque
        rated_tq_file = self.opt_dir + '/rated_tq.txt'
        if os.path.isfile(rated_tq_file):
            f = open(rated_tq_file, "r")
            lines = f.readlines()
            rated_tq = float(lines[0])
            f.close()
        else:
            raise Exception('Could not find rated torque file.')

        # === extract variables === #
        header_len = 1

        if self.training_point_dist == 'linear':
            # determine total number of combinations
            tot_var = 1
            for i in range(0, len(self.sm_var_max)):
                for j in range(0, len(self.sm_var_max[i])):
                    tot_var *= self.sm_var_max[i][j]

        elif self.training_point_dist == 'lhs':

            tot_var = self.num_pts

        else:
            raise Exception('Need to specify training point distribution.')

        # === determine total number of design variables === #
        num_var = 0
        for i in range(0, len(self.sm_var_index)):
            for j in range(0, len(self.sm_var_index[i])):
                num_var += 1

        # === placeholder for var_names === #
        var_names = []

        for i in range(0, num_var):
            var_names.append('var_' + str(i))

        # read in variables
        var_dict = dict()
        for i in range(0, len(var_names)):
            var_dict[var_names[i]] = []

        # open variable .txt file

        if self.training_point_dist == 'linear':

            f = open(self.var_filename, "r")

            # lines = f.readlines(1)
            lines = list(f)

            for i in range(header_len, tot_var+header_len):

                cur_line = lines[i].split()

                # for the case where only varied variables are recorded in sm_var.txt
                if len(cur_line) == num_var+1:
                    for j in range(1, len(cur_line)):
                        var_dict[var_names[j-1]].append(float(cur_line[j]))

            f.close()

            # === extract outputs === #
            # open output .txt file
            f = open(self.DEM_filename, "r")

            # print(list(f))
            lines = list(f)

            # first out line
            first_line = lines[1].split()

            if len(first_line) == 37:
                sgp_range = 17+1
            elif len(first_line) == 17:
                sgp_range = 7+1

            # all outputs (DEMs, loads, tip def) dictionary
            out_dict = dict()
            DEM_names = []

            out_dict['Rootx'] = []
            DEM_names.append('Rootx')
            for i in range(1, sgp_range):
                out_dict['DEMx_' + str(i)] = []
                DEM_names.append('DEMx_' + str(i))

            out_dict['Rooty'] = []
            DEM_names.append('Rooty')
            for i in range(1, sgp_range):
                out_dict['DEMy_' + str(i)] = []
                DEM_names.append('DEMy_' + str(i))

            for i in range(header_len, tot_var + header_len):

                cur_line = lines[i].split()
                for j in range(1, len(cur_line)):
                    out_dict[DEM_names[j - 1]].append(float(cur_line[j]))

            f.close()

        elif self.training_point_dist == 'lhs':

            # === get design variable values / calculated outputs === #
            f_var = open(self.var_filename, "r")
            lines_var = list(f_var)
            f_var.close()

            f_DEM = open(self.DEM_filename, "r")
            lines_DEM = list(f_DEM)
            f_DEM.close()

            f_def = open(self.def_filename, "r")
            lines_def = list(f_def)
            f_def.close()

            f_load = open(self.load_filename, "r")
            lines_load = list(f_load)
            f_load.close()

            # === create var_dict === #

            # iterate for every training point
            for k in range(self.num_pts):

                # for i in range(header_len, tot_var + header_len):
                for i in range(header_len + k, header_len + k + 1):

                    cur_line = lines_var[i].split()

                    # for the case where only varied variables are recorded in sm_var.txt
                    if len(cur_line) == num_var + 1:
                        for j in range(1, len(cur_line)):
                            var_dict[var_names[j - 1]].append(float(cur_line[j]))

            # === DEMs, extreme loads, and tip deflection (oh my!) === #
            # first out line
            first_line = lines_DEM[1].split()

            if len(first_line) == 37:
                sgp_range = 17 + 1
            elif len(first_line) == 17:
                sgp_range = 7 + 1

            # all outputs (DEMs, loads, tip def) dictionary
            out_dict = dict()

            DEM_names = []

            out_dict['Rootx'] = []
            DEM_names.append('Rootx')
            for i in range(1, sgp_range):
                out_dict['DEMx_' + str(i)] = []
                DEM_names.append('DEMx_' + str(i))

            out_dict['Rooty'] = []
            DEM_names.append('Rooty')
            for i in range(1, sgp_range):
                out_dict['DEMy_' + str(i)] = []
                DEM_names.append('DEMy_' + str(i))

            load_names = []

            for i in range(self.nstr):
                out_dict['Edg' + str(i)] = []
                load_names.append('Edg' + str(i))
            for i in range(self.nstr):
                out_dict['Flp' + str(i)] = []
                load_names.append('Flp' + str(i))

            def_names = []

            for i in range(1):
                out_dict['tip' + str(i)] = []
                def_names.append('tip' + str(i))

            for k in range(self.num_pts):
                # DEMx, DEMy

                # for i in range(header_len, tot_var + header_len):
                for i in range(header_len + k, header_len + k + 1):

                    cur_line = lines_DEM[i].split()
                    for j in range(1, len(cur_line)):
                        # out_dict[DEM_names[j - 1]].append(float(cur_line[j]))
                        out_dict[DEM_names[j - 1]].append(float(cur_line[j])*rated_tq)

                # Edg, Flp

                # for i in range(header_len, tot_var + header_len):
                for i in range(header_len + k, header_len + k + 1):

                    cur_line = lines_load[i].split()
                    for j in range(1, len(cur_line)):

                        # out_dict[load_names[j - 1]].append(float(cur_line[j]))
                        out_dict[load_names[j - 1]].append(float(cur_line[j])*rated_tq)

                # tip deflection

                # for i in range(header_len, tot_var + header_len):
                for i in range(header_len + k, header_len + k + 1):

                    cur_line = lines_def[i].split()
                    for j in range(1, len(cur_line)):

                        out_dict[def_names[j - 1]].append(float(cur_line[j])*params['bladeLength'])

        # === Approximation Model === #
        from smt.surrogate_models import QP, LS, KRG, KPLS, KPLSK, RBF

        if self.approximation_model == 'second_order_poly':
            sm_x_fit = QP()
            sm_y_fit = QP()
            sm_x_load_fit = QP()
            sm_y_load_fit = QP()
            sm_def_fit = QP()

            sm_check_fit = QP()

            cv_x_fit = QP()
            cv_y_fit = QP()

        elif self.approximation_model == 'least_squares':
            sm_x_fit = LS()
            sm_y_fit = LS()
            sm_x_load_fit = LS()
            sm_y_load_fit = LS()
            sm_def_fit = LS()

            sm_check_fit = LS()

            cv_x_fit = LS()
            cv_y_fit = LS()

        elif self.approximation_model == 'RBF':

            sm_x_fit = RBF(d0=5)
            sm_y_fit = RBF(d0=5)
            sm_x_load_fit = RBF(d0=5)
            sm_y_load_fit = RBF(d0=5)
            sm_def_fit = RBF(d0=5)

            sm_check_fit = RBF(d0=5)

            cv_x_fit = RBF(d0=5)
            cv_y_fit = RBF(d0=5)

        elif self.approximation_model == 'kriging':

            # initial hyperparameters
            theta0_val = np.zeros([len(self.var_index), 1])
            for i in range(len(theta0_val)):
                theta0_val[i] = self.theta0_val[0]

            sm_x_fit = KRG(theta0=theta0_val)
            sm_y_fit = KRG(theta0=theta0_val)
            sm_x_load_fit = KRG(theta0=theta0_val)
            sm_y_load_fit = KRG(theta0=theta0_val)
            sm_def_fit = KRG(theta0=theta0_val)

            sm_check_fit = KRG(theta0=theta0_val)

            cv_x_fit = KRG(theta0=theta0_val)
            cv_y_fit = KRG(theta0=theta0_val)

        elif self.approximation_model == 'KPLS':
            theta0_val = self.theta0_val

            sm_x_fit = KPLS(theta0=theta0_val)
            sm_y_fit = KPLS(theta0=theta0_val)
            sm_x_load_fit = KPLS(theta0=theta0_val)
            sm_y_load_fit = KPLS(theta0=theta0_val)
            sm_def_fit = KPLS(theta0=theta0_val)

            sm_check_fit = KPLS(theta0=theta0_val)

            cv_x_fit = KPLS(theta0=theta0_val)
            cv_y_fit = KPLS(theta0=theta0_val)

        elif self.approximation_model == 'KPLSK':
            theta0_val = self.theta0_val

            sm_x_fit = KPLSK(theta0=theta0_val)
            sm_y_fit = KPLSK(theta0=theta0_val)
            sm_x_load_fit = KPLSK(theta0=theta0_val)
            sm_y_load_fit = KPLSK(theta0=theta0_val)
            sm_def_fit = KPLSK(theta0=theta0_val)

            sm_check_fit = KPLSK(theta0=theta0_val)

            cv_x_fit = KPLSK(theta0=theta0_val)
            cv_y_fit = KPLSK(theta0=theta0_val)


        else:
            raise Exception('Need to specify which approximation model will be used in surrogate model.')

        # === initialize predicted values === #

        # DEMs
        DEMx_sm = np.zeros([18, 1])
        DEMy_sm = np.zeros([18, 1])

        # loads
        Edg_sm = np.zeros([self.nstr, 1])
        Flp_sm = np.zeros([self.nstr, 1])

        # tip deflection
        def_sm = np.zeros([1, 1])

        # need to get training values: xt -
        num_pts = len(out_dict['Rootx'])
        num_vars = len(var_names)

        xt = np.zeros([num_vars, num_pts])

        # === DEMx_sm, DEMy_sm fit creation === #

        # design variable values; yt - outputs
        yt_x = np.zeros([len(DEMx_sm), num_pts])
        yt_y = np.zeros([len(DEMy_sm), num_pts])

        for i in range(0,len(DEMx_sm)):

            for j in range(0, num_pts):

                # design variable values
                for k in range(0,len(var_names)):

                    # convert from chord / blade length to chord
                    # if var_names == 'chord_sub':
                    if k < 4:
                        xt[k, j] = var_dict[var_names[k]][j] * params['bladeLength']
                    else:
                        xt[k, j] = var_dict[var_names[k]][j]

                # output values
                yt_x[i, j] = out_dict[DEM_names[i]][j]
                yt_y[i, j] = out_dict[DEM_names[i + 18]][j]

        sm_x = sm_x_fit
        sm_x.set_training_values(np.transpose(xt),np.transpose(yt_x))
        sm_x.options['print_global'] = self.print_sm
        sm_x.train()

        sm_y = sm_y_fit
        sm_y.set_training_values(np.transpose(xt),np.transpose(yt_y))
        sm_y.options['print_global'] = self.print_sm
        sm_y.train()

        # === Edg_sm, Flp_sm fit creation === #

        num_pts_load = len(out_dict['Edg0'])

        yt_x_load = np.zeros([len(Edg_sm), num_pts_load])
        yt_y_load = np.zeros([len(Flp_sm), num_pts_load])

        for i in range(0,len(Edg_sm)):

            for j in range(0, num_pts_load):

                # output values
                yt_x_load[i, j] = out_dict[load_names[i]][j]
                yt_y_load[i, j] = out_dict[load_names[i + self.nstr]][j]

        sm_x_load = sm_x_load_fit
        sm_x_load.set_training_values(np.transpose(xt),np.transpose(yt_x_load))
        sm_x_load.options['print_global'] = self.print_sm
        sm_x_load.train()

        sm_y_load = sm_y_load_fit
        sm_y_load.set_training_values(np.transpose(xt),np.transpose(yt_y_load))
        sm_y_load.options['print_global'] = self.print_sm
        sm_y_load.train()

        # === tip deflection fit creation === #

        # num_pts_def = len(out_dict[def_names[0]])
        #
        # yt_def = np.zeros([len(def_sm), num_pts_def])
        #
        # for i in range(0, len(def_sm)):
        #
        #     for j in range(0, num_pts_def):
        #         # output values
        #         yt_def[i, j] = out_dict[def_names[i]][j]
        #
        # sm_def = sm_def_fit
        # sm_def.set_training_values(np.transpose(xt), np.transpose(yt_def))
        # sm_def.options['print_global'] = self.print_sm
        # sm_def.train()

        if self.calc_DEM_using_sm_no_opt:

            # === estimate outputs === #

            # current design variable values
            sv = []
            for i in range(0, len(self.sm_var_names)):

                # chord_sub, theta_sub
                if hasattr(params[self.sm_var_names[i]], '__len__'):
                    for j in range(0, len(self.sm_var_names[i])):

                        if j in self.sm_var_index[i]:

                            # calculate chord / blade length
                            if self.sm_var_names[i] == 'chord_sub':
                                sv.append(params[self.sm_var_names[i]][j] / (params['bladeLength']))
                            else:
                                sv.append(params[self.sm_var_names[i]][j])
                # turbulence intensity
                else:
                    sv.append(params[self.sm_var_names[i]])

            # === predict values === #

            int_sv = np.zeros([len(sv), 1])
            for i in range(0, len(int_sv)):
                int_sv[i] = sv[i]

            # DEMs
            DEMx_sm = np.transpose(sm_x.predict_values(np.transpose(int_sv)))
            DEMy_sm = np.transpose(sm_y.predict_values(np.transpose(int_sv)))

            # extreme loads
            Edg_sm = np.transpose(sm_x_load.predict_values(np.transpose(int_sv)))
            Flp_sm = np.transpose(sm_y_load.predict_values(np.transpose(int_sv)))

            print('Edgewise DEMs:')
            print(DEMx_sm)

            print('Flapwise DEMs:')
            print(DEMy_sm)

            print('Edgewise Extreme Moments:')
            print(Edg_sm)

            print('Flapwise Extreme Moments:')
            print(Flp_sm)

            quit()


        # === created surrogate models to .pkl files
        sm_list = [sm_x, sm_y, sm_x_load, sm_y_load]
        sm_string_list = ['sm_x', 'sm_y', 'sm_x_load', 'sm_y_load']

        for i in range(len(sm_list)):
            pkl_file_name = self.opt_dir + '/' + sm_string_list[i] + '_' + self.approximation_model + '.pkl'
            file_handle = open(pkl_file_name, "w+")
            pickle.dump(sm_list[i], file_handle)

        if self.check_sm_accuracy:

            print('Checking SM accuracy vs. initial design')

            # get design variable values
            # current design variable values

            sv = []
            for i in range(0, len(self.sm_var_names)):

                # chord_sub, theta_sub
                if hasattr(params[self.sm_var_names[i]], '__len__'):
                    for j in range(0, len(self.sm_var_names[i])):

                        if j in self.sm_var_index[i]:

                            # calculate chord / blade length
                            if self.sm_var_names[i] == 'chord_sub':
                                # sv.append( params[self.sm_var_names[i]][j] / ( params['bladeLength']) )
                                sv.append( params[self.sm_var_names[i]][j] / ( 1 ) )
                            else:
                                sv.append(params[self.sm_var_names[i]][j])
                # chord_sub
                else:
                    sv.append(params[self.sm_var_names[i]])

            # === predict values === #

            int_sv = np.zeros([len(sv), 1])
            for i in range(0, len(int_sv)):
                int_sv[i] = sv[i]

            # determine estimated values using surrogate model
            # DEMs
            DEMx_sm = np.transpose(sm_x.predict_values(np.transpose(int_sv)))
            DEMy_sm = np.transpose(sm_y.predict_values(np.transpose(int_sv)))

            # get actual values from .txt file
            f = open(self.opt_dir + '/xDEM_max.txt' )
            # f = open(self.opt_dir + '/xDEM_max_5MW.txt')
            lines = f.readlines()
            DEMx_actual = []
            for i in range(len(lines)):
                DEMx_actual.append(float(lines[i]))
            f.close()

            f = open(self.opt_dir + '/yDEM_max.txt')
            # f = open(self.opt_dir + '/yDEM_max_5MW.txt')
            lines = f.readlines()
            DEMy_actual = []
            for i in range(len(lines)):
                DEMy_actual.append(float(lines[i]))
            f.close()

            DEMx_sm = DEMx_sm.transpose()[0]
            DEMx_actual = np.array(DEMx_actual)

            DEMy_sm = DEMy_sm.transpose()[0]
            DEMy_actual = np.array(DEMy_actual)

            # plot
            turb_class = self.turb_class
            turb_name = 'WP_5MW'
            # turb_name = 'TUM335'

            # DEMx plot
            plt.figure()
            plt.title('DEMx initial design, turbulence: ' + turb_class + ' (surrogate model accuracy)')

            plt.plot(DEMx_actual, '-o', label='Actual')
            plt.plot(DEMx_sm, '--x', label='Estimated')
            plt.xlabel('strain gage position')
            plt.ylabel('N*m')
            plt.xticks(np.linspace(0,17,18))
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMx_comp_' + turb_class + '_' + turb_name + '.png')
            plt.show()

            # DEMy plot
            plt.figure()
            plt.title('DEMy initial design, turbulence: ' + turb_class + ' (surrogate model accuracy)')

            plt.plot(DEMy_actual, '-o', label='Actual')
            plt.plot(DEMy_sm , '--x', label='Estimated')
            plt.xlabel('strain gage position')
            plt.ylabel('N*m')
            plt.xticks(np.linspace(0,17,18))
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMy_comp_' + turb_class + '_' + turb_name + '.png')
            plt.show()

            DEMx_acc = abs(DEMx_actual-DEMx_sm)/DEMx_actual[0]*100.0

            # DEMx accuracy
            plt.figure()
            plt.title('DEMx initial design, turbulence: ' + turb_class + ' (surrogate model accuracy)')

            plt.plot(DEMx_acc, 'o')
            plt.xlabel('strain gage position')
            plt.ylabel('Percent Error (%)')
            plt.xticks(np.linspace(0,17,18))
            plt.savefig(self.dir_saved_plots + '/DEMx_accur_' + turb_class + '_' + turb_name + '.png')
            plt.show()

            DEMy_acc = abs(DEMy_actual-DEMy_sm)/DEMy_actual[0]*100.0

            # DEMy accuracy
            plt.figure()
            plt.title('DEMy initial design, turbulence: ' + turb_class + ' (surrogate model accuracy)')

            plt.plot(DEMy_acc, 'o')
            plt.xlabel('strain gage position')
            plt.ylabel('Percent Error (%)')
            plt.xticks(np.linspace(0,17,18))
            plt.savefig(self.dir_saved_plots + '/DEMy_accur_' + turb_class + '_' + turb_name + '.png')
            plt.show()

            quit()

        if self.check_fit:
            sm = sm_check_fit

            # sm.set_training_values(np.array(var_dict['r_max_chord']), np.array(out_dict['Rooty']))
            sm.set_training_values(np.array(var_dict['var_0']), np.array(out_dict['Rooty']))
            sm.train()

            # predicted value
            val_y = sm.predict_values(np.array(params['r_max_chord']))

            # predicted curve
            num = 100
            fit_x = np.linspace(0.1, 0.5, num)
            fit_y = sm.predict_values(np.array(fit_x))

            plt.figure()
            plt.title('r_max_chord')
            # plt.plot(var_dict['r_max_chord'], out_dict['Rooty'], 'o')
            # plt.plot(var_dict['var_0'], out_dict['Rooty'], 'o')
            plt.plot(params['r_max_chord'], val_y, 'x')
            plt.plot(fit_x, fit_y)
            plt.xlabel('r_max_chord')
            plt.ylabel('Root DEMx (kN*m)')
            # plt.legend(['Training data', 'Calculated Value', 'Prediction'])
            plt.legend(['Calculated Value', 'Prediction'])
            # plt.savefig(self.dir_saved_plots + '/sm_ex.eps')
            plt.savefig(self.dir_saved_plots + '/sm_ex.png')
            plt.show()

            quit()

        # === Do a cross validation, check for total error === #

        if self.do_cv_DEM:

            print('Running DEM cross validation...')

            # === initialize error === #
            DEM_error_x = np.zeros([len(DEMx_sm), self.num_folds])
            percent_DEM_error_x = np.zeros([len(DEMx_sm), self.num_folds])

            DEM_error_y = np.zeros([len(DEMy_sm), self.num_folds])
            percent_DEM_error_y = np.zeros([len(DEMy_sm), self.num_folds])

            for j in range(len(self.kfolds)):

                cur_DEM_error_x = np.zeros([len(DEMx_sm), len(self.kfolds[j])])
                cur_percent_DEM_error_x = np.zeros([len(DEMx_sm), len(self.kfolds[j])])

                cur_DEM_error_y = np.zeros([len(DEMx_sm), len(self.kfolds[j])])
                cur_percent_DEM_error_y = np.zeros([len(DEMx_sm), len(self.kfolds[j])])

                for k in range(len(self.kfolds[j])):

                    cur_kfold = self.kfolds[j]

                    # choose training point indices
                    train_pts = np.linspace(0, self.num_pts - 1, self.num_pts)  # -1 so it's zero-based
                    train_pts = train_pts.tolist()

                    for i in range(0, len(cur_kfold)):
                        train_pts.remove(cur_kfold[i])

                    # choose training point values

                    train_xt = xt[:, train_pts]
                    kfold_xt = xt[:, cur_kfold]

                    train_yt_x = yt_x[:, train_pts]
                    kfold_yt_x = yt_x[:, cur_kfold]

                    train_yt_y = yt_y[:, train_pts]
                    kfold_yt_y = yt_y[:, cur_kfold]

                    # using current design variable values, predict output

                    cv_x = cv_x_fit
                    cv_x.set_training_values(np.transpose(train_xt), np.transpose(train_yt_x))
                    cv_x.options['print_global'] = self.print_sm
                    cv_x.train()

                    cv_y = cv_y_fit
                    cv_y.set_training_values(np.transpose(train_xt), np.transpose(train_yt_y))
                    cv_y.options['print_global'] = self.print_sm
                    cv_y.train()

                    DEMx_cv = np.transpose(cv_x.predict_values(np.array([kfold_xt[:, k]])))
                    DEMy_cv = np.transpose(cv_y.predict_values(np.array([kfold_xt[:, k]])))

                    for i in range(len(DEM_error_x)):
                        cur_DEM_error_x[i][k] = DEMx_cv[i] - kfold_yt_x[:, 0][i]
                        cur_percent_DEM_error_x[i][k] = abs(DEMx_cv[i] - kfold_yt_x[:, k][i]) / kfold_yt_x[:, k][i]

                        cur_DEM_error_y[i][k] = DEMy_cv[i] - kfold_yt_y[:, 0][i]
                        cur_percent_DEM_error_y[i][k] = abs(DEMy_cv[i] - kfold_yt_y[:, k][i]) / kfold_yt_y[:, k][i]

                # average error for specific k-fold
                for i in range(len(DEM_error_x)):
                    DEM_error_x[i][j] = sum(cur_DEM_error_x[i, :]) / len(cur_DEM_error_x[i, :])
                    percent_DEM_error_x[i][j] = sum(cur_percent_DEM_error_x[i, :]) / len(cur_percent_DEM_error_x[i, :])

                    DEM_error_y[i][j] = sum(cur_DEM_error_y[i, :]) / len(cur_DEM_error_y[i, :])
                    percent_DEM_error_y[i][j] = sum(cur_percent_DEM_error_y[i, :]) / len(cur_percent_DEM_error_y[i, :])

            # average percent error over all k-folds
            avg_percent_DEM_error_x = np.zeros([len(DEM_error_x), 1])
            avg_percent_DEM_error_y = np.zeros([len(DEM_error_y), 1])

            rms_percent_DEM_error_x = np.zeros([len(DEM_error_x), 1])
            rms_percent_DEM_error_y = np.zeros([len(DEM_error_y), 1])

            # calculate root mean square error
            for i in range(len(DEM_error_x)):

                avg_percent_DEM_error_x[i] = sum(percent_DEM_error_x[i, :]) / len(percent_DEM_error_x[i, :])
                avg_percent_DEM_error_y[i] = sum(percent_DEM_error_y[i, :]) / len(percent_DEM_error_y[i, :])

                squared_total_x = 0.0
                squared_total_y = 0.0
                for index in range(len(percent_DEM_error_x[i, :])):
                    squared_total_x += percent_DEM_error_x[i, index] ** 2.0
                    squared_total_y += percent_DEM_error_y[i, index] ** 2.0

                rms_percent_DEM_error_x[i] = (squared_total_x / len(percent_DEM_error_x[i, :])) ** 0.5
                rms_percent_DEM_error_y[i] = (squared_total_y / len(percent_DEM_error_y[i, :])) ** 0.5

            # maximum percent error over all k-folds
            max_percent_DEM_error_x = np.zeros([len(DEM_error_x), 1])
            max_percent_DEM_error_y = np.zeros([len(DEM_error_y), 1])

            for i in range(len(DEM_error_x)):
                max_percent_DEM_error_x[i] = max(percent_DEM_error_x[i, :])
                max_percent_DEM_error_y[i] = max(percent_DEM_error_y[i, :])

            # root mean square error over all DEMx, DEMy points
            total_squared_total_x = 0.0
            total_squared_total_y = 0.0
            for index in range(len(rms_percent_DEM_error_x)):
                total_squared_total_x += rms_percent_DEM_error_x[index] ** 2.0
                total_squared_total_y += rms_percent_DEM_error_y[index] ** 2.0

            rms_DEM_error_x = (total_squared_total_x / len(rms_percent_DEM_error_x)) ** 0.5
            rms_DEM_error_y = (total_squared_total_y / len(rms_percent_DEM_error_y)) ** 0.5

            # root mean square error overall
            rms_error = ( (rms_DEM_error_x ** 2.0 + rms_DEM_error_y ** 2.0) / 2.0) ** 0.5


            print('rms_error')
            print(rms_error)
            # quit()


            # save error values in .txt file
            error_file_name = str(self.opt_dir) + '/error_' + self.approximation_model + '_' + str(
                self.num_pts) + '.txt'
            ferror = open(error_file_name, "w+")
            ferror.write(str(rms_DEM_error_x[0]) + '\n')
            ferror.write(str(rms_DEM_error_y[0]))
            ferror.close()

            # DEMx plot
            plt.figure()
            plt.title('DEMx k-fold check (surrogate model accuracy)')

            plt.plot(avg_percent_DEM_error_x * 100.0, 'x', label='avg error')
            plt.plot(max_percent_DEM_error_x * 100.0, 'o', label='max error')
            plt.xlabel('strain gage position')
            plt.ylabel('model accuracy (%)')
            plt.xticks(np.linspace(0,17,18))
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMx_kfold_' + self.turb_class + '.png')
            plt.show()

            # DEMx plot
            plt.figure()
            plt.title('DEMy k-fold check (surrogate model accuracy)')

            plt.plot(avg_percent_DEM_error_y * 100.0, 'x', label='avg error')
            plt.plot(max_percent_DEM_error_y * 100.0, 'o', label='max error')
            plt.xlabel('strain gage position')
            plt.ylabel('model accuracy (%)')
            plt.xticks(np.linspace(0,17,18))
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMy_kfold_' + self.turb_class + '.png')
            plt.show()

            print(avg_percent_DEM_error_x)
            print(max_percent_DEM_error_x)

            print(avg_percent_DEM_error_y)
            print(max_percent_DEM_error_y)

            quit()

        if self.do_cv_Load:

            print('Running extreme loads cross validation...')

            # === initialize error === #
            Edg_error = np.zeros([len(Edg_sm), self.num_folds])
            percent_Edg_error = np.zeros([len(Edg_sm), self.num_folds])

            Flp_error = np.zeros([len(Flp_sm), self.num_folds])
            percent_Flp_error = np.zeros([len(Flp_sm), self.num_folds])

            for j in range(len(self.kfolds)):

                cur_Edg_error = np.zeros([len(Edg_sm), len(self.kfolds[j])])
                cur_percent_Edg_error = np.zeros([len(Edg_sm), len(self.kfolds[j])])

                cur_Flp_error = np.zeros([len(Flp_sm), len(self.kfolds[j])])
                cur_percent_Flp_error = np.zeros([len(Flp_sm), len(self.kfolds[j])])

                for k in range(len(self.kfolds[j])):

                    cur_kfold = self.kfolds[j]

                    # choose training point indices
                    train_pts = np.linspace(0, self.num_pts - 1, self.num_pts)  # -1 so it's zero-based
                    train_pts = train_pts.tolist()

                    for i in range(0, len(cur_kfold)):
                        train_pts.remove(cur_kfold[i])

                    # choose training point values
                    train_xt = xt[:, train_pts]
                    kfold_xt = xt[:, cur_kfold]

                    train_yt_x = yt_x_load[:, train_pts]
                    kfold_yt_x = yt_x_load[:, cur_kfold]

                    train_yt_y = yt_y_load[:, train_pts]
                    kfold_yt_y = yt_y_load[:, cur_kfold]

                    # using current design variable values, predict output

                    cv_x = cv_x_fit
                    cv_x.set_training_values(np.transpose(train_xt), np.transpose(train_yt_x))
                    cv_x.options['print_global'] = self.print_sm
                    cv_x.train()

                    cv_y = cv_y_fit
                    cv_y.set_training_values(np.transpose(train_xt), np.transpose(train_yt_y))
                    cv_y.options['print_global'] = self.print_sm
                    cv_y.train()

                    Edg_cv = np.transpose(cv_x.predict_values(np.array([kfold_xt[:, k]])))
                    Flp_cv = np.transpose(cv_y.predict_values(np.array([kfold_xt[:, k]])))

                    for i in range(len(Edg_error)):
                        cur_Edg_error[i][k] = Edg_cv[i] - kfold_yt_x[:, 0][i]
                        cur_percent_Edg_error[i][k] = abs(Edg_cv[i] - kfold_yt_x[:, k][i]) / kfold_yt_x[:, k][i]

                        cur_Flp_error[i][k] = Flp_cv[i] - kfold_yt_y[:, 0][i]
                        cur_percent_Flp_error[i][k] = abs(Flp_cv[i] - kfold_yt_y[:, k][i]) / kfold_yt_y[:, k][i]

                # average error for specific k-fold
                for i in range(len(Edg_error)):
                    Edg_error[i][j] = sum(cur_Edg_error[i, :]) / len(cur_Edg_error[i, :])
                    percent_Edg_error[i][j] = sum(cur_percent_Edg_error[i, :]) / len(cur_percent_Edg_error[i, :])

                    Flp_error[i][j] = sum(cur_Flp_error[i, :]) / len(cur_Flp_error[i, :])
                    percent_Flp_error[i][j] = sum(cur_percent_Flp_error[i, :]) / len(cur_percent_Flp_error[i, :])

            # average percent error over all k-folds
            avg_percent_Edg_error = np.zeros([len(Edg_error), 1])
            avg_percent_Flp_error = np.zeros([len(Flp_error), 1])

            rms_percent_Edg_error = np.zeros([len(Edg_error), 1])
            rms_percent_Flp_error = np.zeros([len(Flp_error), 1])

            # calculate root mean square error
            for i in range(len(Edg_error)):

                avg_percent_Edg_error[i] = sum(percent_Edg_error[i, :]) / len(percent_Edg_error[i, :])
                avg_percent_Flp_error[i] = sum(percent_Flp_error[i, :]) / len(percent_Flp_error[i, :])

                squared_total_x = 0.0
                squared_total_y = 0.0
                for index in range(len(percent_Edg_error[i, :])):
                    squared_total_x += percent_Edg_error[i, index] ** 2.0
                    squared_total_y += percent_Flp_error[i, index] ** 2.0

                rms_percent_Edg_error[i] = (squared_total_x / len(percent_Edg_error[i, :])) ** 0.5
                rms_percent_Flp_error[i] = (squared_total_y / len(percent_Flp_error[i, :])) ** 0.5

            # maximum percent error over all k-folds
            max_percent_Edg_error = np.zeros([len(Edg_error), 1])
            max_percent_Flp_error = np.zeros([len(Flp_error), 1])

            for i in range(len(Edg_error)):
                max_percent_Edg_error[i] = max(percent_Edg_error[i, :])
                max_percent_Flp_error[i] = max(percent_Flp_error[i, :])

            # root mean square error over all DEMx, DEMy points
            total_squared_total_x = 0.0
            total_squared_total_y = 0.0
            for index in range(len(rms_percent_Edg_error)):
                total_squared_total_x += rms_percent_Edg_error[index] ** 2.0
                total_squared_total_y += rms_percent_Flp_error[index] ** 2.0

            rms_Edg_error = (total_squared_total_x / len(rms_percent_Edg_error)) ** 0.5
            rms_Flp_error = (total_squared_total_y / len(rms_percent_Flp_error)) ** 0.5

            # root mean square error overall
            rms_error = (((rms_Edg_error ** 2.0 + rms_Flp_error ** 2.0)) / 2.0) ** 0.5

            print('rms_error')
            print(rms_error)
            # quit()


            # save error values in .txt file
            error_file_name = str(self.opt_dir) + '/error_' + self.approximation_model + '_' + str(
                self.num_pts) + '.txt'
            ferror = open(error_file_name, "w+")
            ferror.write(str(rms_Edg_error[0]) + '\n')
            ferror.write(str(rms_Flp_error[0]))
            ferror.close()

            # DEMx plot
            plt.figure()
            plt.title('DEMx k-fold check (surrogate model accuracy)')

            plt.plot(avg_percent_Edg_error * 100.0, 'x', label='avg error')
            plt.plot(max_percent_Edg_error * 100.0, 'o', label='max error')
            plt.xlabel('strain gage position')
            plt.ylabel('model accuracy (%)')
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMx_kfold.png')
            plt.show()

            # DEMx plot
            plt.figure()
            plt.title('DEMy k-fold check (surrogate model accuracy)')

            plt.plot(avg_percent_Flp_error * 100.0, 'x', label='avg error')
            plt.plot(max_percent_Flp_error * 100.0, 'o', label='max error')
            plt.xlabel('strain gage position')
            plt.ylabel('model accuracy (%)')
            plt.legend()
            plt.savefig(self.dir_saved_plots + '/DEMy_kfold.png')
            plt.show()

            quit()

        # if self.do_cv_def:
        #
        #     print('Running tip deflection cross validation...')
        #
        #     # === initialize error === #
        #     def_error = np.zeros([len(def_sm), self.num_folds])
        #     percent_def_error = np.zeros([len(def_sm), self.num_folds])
        #
        #     for j in range(len(self.kfolds)):
        #
        #         cur_def_error = np.zeros([len(def_sm), len(self.kfolds[j])])
        #         cur_percent_def_error = np.zeros([len(def_sm), len(self.kfolds[j])])
        #
        #         for k in range(len(self.kfolds[j])):
        #
        #             cur_kfold = self.kfolds[j]
        #
        #             # choose training point indices
        #             train_pts = np.linspace(0, self.num_pts - 1, self.num_pts)  # -1 so it's zero-based
        #             train_pts = train_pts.tolist()
        #
        #             for i in range(0, len(cur_kfold)):
        #                 train_pts.remove(cur_kfold[i])
        #
        #             # choose training point values
        #             train_xt = xt[:, train_pts]
        #             kfold_xt = xt[:, cur_kfold]
        #
        #             train_yt_x = yt_def[:, train_pts]
        #             kfold_yt_x = yt_def[:, cur_kfold]
        #
        #             # using current design variable values, predict output
        #
        #             cv_x = cv_x_fit
        #             cv_x.set_training_values(np.transpose(train_xt), np.transpose(train_yt_x))
        #             cv_x.options['print_global'] = self.print_sm
        #             cv_x.train()
        #
        #             def_cv = np.transpose(cv_x.predict_values(np.array([kfold_xt[:, k]])))
        #
        #             for i in range(len(def_error)):
        #                 cur_def_error[i][k] = def_cv[i] - kfold_yt_x[:, 0][i]
        #                 cur_percent_def_error[i][k] = abs(def_cv[i] - kfold_yt_x[:, k][i]) / kfold_yt_x[:, k][i]
        #
        #         # average error for specific k-fold
        #         for i in range(len(def_error)):
        #             def_error[i][j] = sum(cur_def_error[i, :]) / len(cur_def_error[i, :])
        #             percent_def_error[i][j] = sum(cur_percent_def_error[i, :]) / len(cur_percent_def_error[i, :])
        #
        #     # average percent error over all k-folds
        #     avg_percent_def_error = np.zeros([len(def_error), 1])
        #
        #     rms_percent_def_error = np.zeros([len(def_error), 1])
        #
        #     # calculate root mean square error
        #     for i in range(len(def_error)):
        #
        #         avg_percent_def_error[i] = sum(percent_def_error[i, :]) / len(percent_def_error[i, :])
        #
        #         squared_total_x = 0.0
        #         for index in range(len(percent_def_error[i, :])):
        #             squared_total_x += percent_def_error[i, index] ** 2.0
        #
        #         rms_percent_def_error[i] = (squared_total_x / len(percent_def_error[i, :])) ** 0.5
        #
        #     # maximum percent error over all k-folds
        #     max_percent_def_error = np.zeros([len(def_error), 1])
        #
        #     for i in range(len(def_error)):
        #         max_percent_def_error[i] = max(percent_def_error[i, :])
        #
        #     # root mean square error over all DEMx, DEMy points
        #     total_squared_total_x = 0.0
        #     for index in range(len(rms_percent_def_error)):
        #         total_squared_total_x += rms_percent_def_error[index] ** 2.0
        #
        #     rms_def_error = (total_squared_total_x / len(rms_percent_def_error)) ** 0.5
        #
        #     # root mean square error overall
        #     rms_error = rms_def_error
        #
        #     print('rms_error')
        #     print(rms_error)
        #     quit()


class Calculate_FAST_sm_training_points(Component):
    def __init__(self, FASTinfo, naero, nstr):
        super(Calculate_FAST_sm_training_points, self).__init__()

        # === design variables === #
        self.add_param('r_max_chord', val=0.0)
        self.add_param('chord_sub',  val=np.zeros(4))
        self.add_param('theta_sub', val=np.zeros(4))
        self.add_param('sparT', val=np.zeros(5))
        self.add_param('teT', val=np.zeros(5))

        # === outputs from createFASTconstraints === #
        self.add_param('DEMx', shape=18, desc='DEMx')
        self.add_param('DEMy', shape=18, desc='DEMy')

        self.add_param('Edg_max', shape=nstr, desc='FAST Edg_max')
        self.add_param('Flp_max', shape=nstr, desc='FAST Flp_max')

        self.add_param('max_tip_def', shape=1, desc='FAST calculated maximum tip deflection')

        # === surrogate model options === #
        self.FASTinfo = FASTinfo

        self.training_point_dist = FASTinfo['training_point_dist'] # 'lhs', 'linear'

        self.sm_var_spec = FASTinfo['sm_var_spec']
        self.sm_var_index = FASTinfo['sm_var_index']
        self.sm_var_names = FASTinfo['sm_var_names']

        if self.training_point_dist == 'lhs':
            self.num_pts = FASTinfo['num_pts']

        else:
            self.sm_var_max = FASTinfo['sm_var_max']

        self.sm_var_file = FASTinfo['sm_var_file']
        self.sm_DEM_file = FASTinfo['sm_DEM_file']
        self.sm_load_file = FASTinfo['sm_load_file']
        self.sm_def_file = FASTinfo['sm_def_file']

        self.opt_dir = FASTinfo['opt_dir']

        self.var_filename = self.opt_dir + '/' + self.sm_var_file
        self.DEM_filename = self.opt_dir + '/' + self.sm_DEM_file
        self.load_filename = self.opt_dir + '/' + self.sm_load_file
        self.def_filename = self.opt_dir + '/' + self.sm_def_file

        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd']

        total_num_bl_gages = 0
        for i in range(0, len(self.NBlGages)):
            total_num_bl_gages += self.NBlGages[i]

        self.add_param('bladeLength', shape=1, desc='Blade length')
        self.add_param('nBlades', val=3, desc='Number of Blades')

        # other surrogate model inputs
        self.add_param('turbulence_class', val=Enum('A', 'B', 'C'), desc='IEC turbulence class', pass_by_obj=True)
        self.add_param('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True)
        self.add_param('turbulence_intensity', val=0.14, desc='IEC turbine class intensity', pass_by_obj=True)
        self.add_param('af_idx', val=np.zeros(naero))
        self.add_param('airfoil_types', val=np.zeros(8))

    def solve_nonlinear(self, params, unknowns, resids):

        # get turbulence intensity value
        if params['turbulence_class'] == 'A':
            params['turbulence_intensity'] = 0.12
        elif params['turbulence_class'] == 'B':
            params['turbulence_intensity'] = 0.14
        elif params['turbulence_class'] == 'C':
            params['turbulence_intensity'] = 0.16

        # get rated torque
        rated_tq_file = self.opt_dir + '/rated_tq.txt'
        if os.path.isfile(rated_tq_file):
            f = open(rated_tq_file, "r")
            lines = f.readlines()
            rated_tq = float(lines[0])
            f.close()
        else:
            raise Exception('Could not find rated torque file.')

        def replace_line(file_name, line_num, text):
            lines = open(file_name, 'r').readlines()
            lines[line_num] = text
            out = open(file_name, 'w')
            out.writelines(lines)
            out.close()

        # === variable and output files === #

        # if training points are laid out linearly
        if self.training_point_dist == 'linear':

            # total variations
            tv = []
            for i in range(0, len(self.sm_var_max)):
                for j in range(0, len(self.sm_var_max[i])):
                    tv.append(self.sm_var_max[i][j])

            # specific variation
            sv = []
            for i in range(0, len(self.sm_var_spec)):
                for j in range(0, len(self.sm_var_spec[i])):
                    sv.append(self.sm_var_spec[i][j])

            # check if output file exists (if it doesn't, create it)
            if not (os.path.isfile(self.DEM_filename)):
                # create file
                f = open(self.DEM_filename,"w+")

                # write a header
                header0 = 'variable points: '
                for i in range(0, len(self.sm_var_names)):

                    header0 += self.sm_var_names[i]
                    for j in range(0, len(self.sm_var_index[i])):
                        header0 += '_' + str(self.sm_var_index[i][j])

                    header0 += ' '

                    for j in range(0, len(self.sm_var_spec[i])):
                        header0 += str(self.sm_var_max[i][j]) + ' '

                f.write(header0+'\n')

                # total variation product
                n_tv = np.prod(tv)

                for i in range(0,n_tv):
                    f.write('-- place holder --'+'\n')

                f.close()

            # variable file
            if not (os.path.isfile(self.var_filename)):
                # create file
                f = open(self.var_filename, "w+")

                # write a header
                header0 = 'variable points: '
                for i in range(0, len(self.sm_var_names)):

                    header0 += self.sm_var_names[i]
                    for j in range(0, len(self.sm_var_index[i])):
                        header0 += '_' + str(self.sm_var_index[i][j])

                    header0 += ' '

                    for j in range(0, len(self.sm_var_spec[i])):
                        header0 += str(self.sm_var_max[i][j]) + ' '

                f.write(header0 + '\n')

                # total variation product
                n_tv = np.prod(tv)

                for i in range(0, n_tv):
                    f.write('-- place holder --' + '\n')

                f.close()

            # determine which line we should write to

            # get position
            def surr_model_pos(spec_var, max_var):
                pos = 0
                for i in range(0, len(max_var) - 1):
                    pos += (spec_var[i] - 1) * np.prod(max_var[i + 1:len(max_var)])
                pos += spec_var[len(spec_var) - 1] - 1

                return pos

            spec_pos = surr_model_pos(sv, tv)

            header_len = 1

            # write first entry to line as naming convention (ex. 1_2_2_0_1 if 5 variables are being used)
            DEM_text = 'var_'
            for i in range(0, len(sv)):
                DEM_text += str(sv[i])+'_'

            # put DEMx and DEMy as values on line
            for i in range(0, len(params['DEMx'])):
                DEM_text += ' ' + str(params['DEMx'][i])

            for i in range(0, len(params['DEMy'])):
                DEM_text += ' ' + str(params['DEMy'][i])

            replace_line(self.DEM_filename, spec_pos+header_len, DEM_text+'\n')

            # add for var_file
            var_text = 'var_'
            for i in range(0, len(sv)):
                var_text += str(sv[i])+'_'

            for i in range(0, len(self.sm_var_names)):

                # chord_sub, theta_sub
                if hasattr(params[self.sm_var_names[i]],'__len__'):
                    for j in range(0, len(params[self.sm_var_names[i]])):
                        if j in self.sm_var_index[i]:

                            # nondimensionalize chord_sub (replace c with c/r)
                            if self.sm_var_names[i] == 'chord_sub':
                                var_text += ' ' + str(params[self.sm_var_names[i]][j]/params['bladeLength'])
                            else:
                                var_text += ' ' + str(params[self.sm_var_names[i]][j])

                # r_max_chord
                else:
                    var_text += ' ' + str(params[self.sm_var_names[i]])

            replace_line(self.var_filename, spec_pos + header_len, var_text + '\n')

        # if training points are determined with latin hypercube sampling
        elif self.training_point_dist == 'lhs':

            header_len = 1
            # === initialize variable file === #

            if not (os.path.isfile(self.var_filename)):
                # create file

                f = open(self.var_filename, "w+")

                # header line
                header0 = 'variable points: '
                for i in range(0, len(self.sm_var_names)):

                    header0 += self.sm_var_names[i]
                    for j in range(0, len(self.sm_var_index[i])):
                        header0 += '_' + str(self.sm_var_index[i][j])

                    header0 += ' '

                f.write(header0 + '\n')

                for i in range(0, self.num_pts):
                    f.write('-- place holder --' + '\n')

                f.close

            # === add set input variables to variable file === #

            var_text = 'num_pt_' + str(self.sm_var_spec)

            for i in range(0, len(self.sm_var_names)):

                # chord_sub, theta_sub
                if hasattr(params[self.sm_var_names[i]], '__len__'):
                    for j in range(0, len(params[self.sm_var_names[i]])):
                        if j in self.sm_var_index[i]:

                            # calculate chord / blade length
                            if self.sm_var_names[i] == 'chord_sub':
                                var_text += ' ' + str( params[self.sm_var_names[i]][j] / ( params['bladeLength'] ) )
                            else:
                                var_text += ' ' + str(params[self.sm_var_names[i]][j])

                # turbulence
                else:
                    var_text += ' ' + str(params[self.sm_var_names[i]])

            # === create output files, initialize lines === #

            # output, load, and def files
            file_list = [self.DEM_filename, self.load_filename, self.def_filename]

            for k in range(len(file_list)):
                if not (os.path.isfile(file_list[k])):
                    # create file
                    f = open(file_list[k], "w+")

                    # write a header
                    header0 = 'variable points: '
                    for i in range(0, len(self.sm_var_names)):

                        header0 += self.sm_var_names[i]
                        for j in range(0, len(self.sm_var_index[i])):
                            header0 += '_' + str(self.sm_var_index[i][j])

                        header0 += ' '

                    f.write(header0 + '\n')

                    for i in range(0, self.num_pts):
                        f.write('-- place holder --' + '\n')

                    f.close()

            # === create line for DEM output file === #

            DEM_text = 'pt_' + str(self.sm_var_spec)

            # put DEMx and DEMy as values on line
            for i in range(0, len(params['DEMx'])):
                # DEM_text += ' ' + str(params['DEMx'][i])
                DEM_text += ' ' + str(params['DEMx'][i]/rated_tq)

            for i in range(0, len(params['DEMy'])):
                # DEM_text += ' ' + str(params['DEMy'][i])
                DEM_text += ' ' + str(params['DEMy'][i]/rated_tq)

            # === create line for extreme load output file === #

            load_text = 'pt_' + str(self.sm_var_spec)

            # put Edg_max and Flp_max as values on line
            for i in range(0, len(params['Edg_max'])):
                # load_text += ' ' + str(params['Edg_max'][i])
                load_text += ' ' + str(params['Edg_max'][i]/rated_tq)

            for i in range(0, len(params['Flp_max'])):
                # load_text += ' ' + str(params['Flp_max'][i])
                load_text += ' ' + str(params['Flp_max'][i]/rated_tq)

            # === create line for extreme deflection output file === #
            def_text = 'pt_' + str(self.sm_var_spec)

            def_text += ' ' + str(params['max_tip_def']/params['bladeLength'])

            # === write to all files === #
            replace_line(self.var_filename, self.sm_var_spec + header_len, var_text + '\n')

            replace_line(self.DEM_filename, self.sm_var_spec + header_len, DEM_text + '\n')

            replace_line(self.load_filename, self.sm_var_spec + header_len, load_text + '\n')

            replace_line(self.def_filename, self.sm_var_spec + header_len, def_text + '\n')

        else:
            raise Exception('Need to specify training point distribution.')

        return


class CreateFASTConstraints(Component):
    def __init__(self, naero, nstr, FASTinfo, WNDfile_List, caseids):
        super(CreateFASTConstraints, self).__init__()

        self.caseids = caseids
        self.WNDfile_List = WNDfile_List
        self.dT = FASTinfo['dT']
        self.description = FASTinfo['description']
        self.path = FASTinfo['path']
        self.opt_dir = FASTinfo['opt_dir']

        self.train_sm = FASTinfo['train_sm']
        if self.train_sm:
            self.sm_dir = FASTinfo['sm_dir']

        self.NBlGages = FASTinfo['NBlGages']
        self.BldGagNd = FASTinfo['BldGagNd']
        self.Run_Once = FASTinfo['Run_Once']

        self.dir_saved_plots = FASTinfo['dir_saved_plots']

        self.check_results = FASTinfo['check_results']
        self.check_sgp_spline = FASTinfo['check_sgp_spline']
        self.check_peaks = FASTinfo['check_peaks']
        self.check_rainflow = FASTinfo['check_rainflow']
        self.check_rm_time = FASTinfo['check_rm_time']

        # only works if check_damage is also set as 'true'
        self.check_nom_DEM_damage = FASTinfo['check_nom_DEM_damage']

        self.sgp = FASTinfo['sgp']

        self.wndfiletype = FASTinfo['wnd_type_list']
        self.Tmax_turb = FASTinfo['Tmax_turb']
        self.Tmax_nonturb = FASTinfo['Tmax_nonturb']
        self.turb_sf = FASTinfo['turb_sf']
        self.rm_time = FASTinfo['rm_time']

        self.save_tq = FASTinfo['save_rated_torque']
        self.DLC_List = FASTinfo['DLC_List']

        self.m_value = FASTinfo['m_value']

        self.add_param('cfg_master', val=dict(), pass_by_obj=False)

        self.add_param('rstar_damage', shape=naero + 1, desc='nondimensional radial locations of damage equivalent moments')

        self.add_param('initial_str_grid', shape=nstr, desc='initial structural grid on unit radius')
        self.add_param('initial_aero_grid', shape=naero, desc='initial structural grid on unit radius')

        for i in range(0, len(caseids)):
            self.add_param(caseids[i], val=dict())

        # DEMs
        self.add_output('DEMx', val=np.zeros(naero+1))
        self.add_output('DEMy', val=np.zeros(naero+1))

        # Tip Deflection
        self.add_output('max_tip_def', val=0.0)

        # Structure
        self.eme_fit = FASTinfo['eme_fit']
        self.add_output('Edg_max', val=np.zeros(nstr))
        self.add_output('Flp_max', val=np.zeros(nstr))

    def solve_nonlinear(self, params, unknowns, resids):

        # === Check Results === #
        resultsdict = params[self.caseids[0]]
        if self.check_results:

            bm_param = ['RootMyb1', 'OoPDefl1', 'GenTq', 'RotThrust', 'RotTorq', 'Spn3MLxb1', 'RotPwr', 'GenPwr', 'RootFxc1']
            bm_param_units = ['kN*m', 'm', 'kN*m', 'kN', 'kN*m', 'kN*m', 'kW', 'kW', 'kN']

            # save certain values
            for i in range(0, len(bm_param)):
                f = open('plots/data_files/' + bm_param[i] + '.txt', "w+")
                for j in range(len(resultsdict[bm_param[i]])):
                    f.write(str(resultsdict[bm_param[i]][j]) + '\n')
                f.close()

            # bm_param = ['GenPwr']
            # bm_param_units = ['kW']

            # bm_param = ['RootMyb1', 'OoPDefl1', 'Spn3MLxb1']
            # bm_param_units = ['kN*m', 'm', 'kN*m']

            # for i in range(0, len(bm_param)):
                # f = open(bm_param[i]+'.txt',"w+")
                # for j in range(len(resultsdict[bm_param[i]])):
                #     f.write(str(resultsdict[bm_param[i]][j])+'\n')
                # f.close()

            for i in range(0, len(bm_param)):
            #     f = open(bm_param[i]+'.txt',"r")
            #     lines = f.readlines()
            #     plot_param = []
            #     for j in range(len(lines)):
            #         plot_param.append(float(lines[j]))
            #     f.close()

                plt.figure()
                plt.plot(resultsdict[bm_param[i]])
                # plt.plot(plot_param, '--', label = 'Nonvaried Values')
                plt.xlabel('Simulation Step')
                plt.ylabel(bm_param[i] + ' (' + bm_param_units[i] + ')')
                plt.title(bm_param[i])
                # plt.legend()
                plt.savefig(self.dir_saved_plots + '/plots/param_plots/' + bm_param[i] + '_test.png')

                plt.show()

            quit()

        # === save rated torque === #
        if self.save_tq:
            if self.DLC_List[0] == 'DLC_0_0':

                print('Calculating rated torque...')

                gen_tq_avg = sum(resultsdict['GenTq'])/len(resultsdict['GenTq'])

                rated_tq_file = self.opt_dir + '/rated_tq.txt'

                if os.path.isfile((rated_tq_file)):
                    print('Rated Tq file already created.')
                else:
                    f = open(rated_tq_file, "w+")
                    f.write(str(gen_tq_avg))
                    f.close()

            else:
                raise Exception('Need to specify DLC_0_0 as first DLC if you want to save rated torque.')

        # total number of virtual strain gages
        tot_BldGagNd = []
        for i in range(0, len(self.BldGagNd)):
            for j in range(0, self.NBlGages[i]):
                tot_BldGagNd.append(self.BldGagNd[i][j])

        total_num_bl_gages = 0
        max_gage = 0
        for i in range(0, len(self.NBlGages)):
            total_num_bl_gages += self.NBlGages[i]
            max_gage = max(max_gage, max(self.BldGagNd[i]))

        # === DEM / structural calculations === #

        DEMx_master_array = np.zeros([len(self.WNDfile_List), 1 + max_gage])
        DEMy_master_array = np.zeros([len(self.WNDfile_List), 1 + max_gage])

        # maxes of DEMx, DEMy, (will be promoted)
        DEMx_max = np.zeros([1 + total_num_bl_gages, 1])
        DEMy_max = np.zeros([1 + total_num_bl_gages, 1])

        Edg_max_array = np.zeros([len(self.WNDfile_List), 1 + max_gage])
        Flp_max_array = np.zeros([len(self.WNDfile_List), 1 + max_gage])

        # maxes of Edg, Flp, (will be promoted)
        Edg_max = np.zeros([1 + total_num_bl_gages, 1])
        Flp_max = np.zeros([1 + total_num_bl_gages, 1])

        # === extrapolated loads variables === #
        # peaks master

        peaks_wnd_x = dict()
        for j in range(len(self.WNDfile_List)):
            peaks_wnd_x[str(j+1)] = dict()
            peaks_wnd_x[str(j+1)]['root'] = []
            for i in range(0, total_num_bl_gages):
                peaks_wnd_x[str(j+1)]['bld_gage_' + str(tot_BldGagNd[i])] = []

        peaks_wnd_y = dict()
        for j in range(len(self.WNDfile_List)):
            peaks_wnd_y[str(j+1)] = dict()
            peaks_wnd_y[str(j+1)]['root'] = []
            for i in range(0, total_num_bl_gages):
                peaks_wnd_y[str(j+1)]['bld_gage_' + str(tot_BldGagNd[i])] = []

        # peaks max (will be promoted)
        peaks_max_x = dict()
        peaks_max_x['root'] = []
        for i in range(0, total_num_bl_gages):
            peaks_max_x['bld_gage_' + str(tot_BldGagNd[i])] = []

        peaks_max_y = dict()
        peaks_max_y['root'] = []
        for i in range(0, total_num_bl_gages):
            peaks_max_y['bld_gage_' + str(tot_BldGagNd[i])] = []

        # === cycle through each set of strain gages (k) and each wind file (i) === #
        if self.train_sm:
            FAST_opt_dir = self.sm_dir
        else:
            FAST_opt_dir = self.opt_dir

        for k in range(0, len(self.NBlGages)):

            for i in range(0 + 1, len(self.WNDfile_List) + 1):

                spec_caseid = k*len(self.WNDfile_List) + i - 1

                # === extrapolated loads variables === #
                # peaks master
                peaks_master_x = dict()
                peaks_master_x['root'] = []
                for i_index in range(0, total_num_bl_gages):
                    peaks_master_x['bld_gage_' + str(tot_BldGagNd[i_index])] = []

                peaks_master_y = dict()
                peaks_master_y['root'] = []
                for i_index in range(0, total_num_bl_gages):
                    peaks_master_y['bld_gage_' + str(tot_BldGagNd[i_index])] = []

                # === naming conventions === #

                spec_caseid = k*len(self.WNDfile_List)+(i-1)
                resultsdict = params[self.caseids[spec_caseid]]

                FAST_wnd_directory = FAST_opt_dir + '/' + 'sgp' + str(self.sgp[k]) + '/' + self.caseids[spec_caseid]

                # === rainflow calculation files === #

                # files = [FAST_wnd_directory + '/fst_runfile.outb']
                files = [FAST_wnd_directory + '/fst_runfile.out']

                # read titles of file, since they don't seem to be in order
                file_rainflow = open(files[0])
                line_rainflow = file_rainflow.readlines()

                # extract names fron non-binary FAST output file
                name_line = 6
                str_val = re.findall("\w+", line_rainflow[name_line])

                # create output_array (needed for rainflow calculation)
                output_array = []

                # make RootMxb1 first in output_array
                for j in range (0,len(str_val)):
                    if str_val[j] == 'RootMxb1':
                        output_array.append(j)

                # make Spn1MLxb1 next in output array
                for l in range(0,self.NBlGages[k]):
                    for j in range(0,len(str_val)):
                        if str_val[j] == 'Spn{0}MLxb1'.format(str(l+1)):
                            output_array.append(j)

                # make RootMyb1 next in output_array
                for j in range(0, len(str_val)):
                    if str_val[j] == 'RootMyb1':
                        output_array.append(j)

                # make Spn1MLyb1 next in output array
                for l in range(0, self.NBlGages[k]):
                    for j in range(0, len(str_val)):
                        if str_val[j] == 'Spn{0}MLyb1'.format(str(l+1)):
                            output_array.append(j)

                # === perform rainflow calculations === #
                from rainflow import do_rainflow

                SNslope = np.zeros([1,len(output_array)])
                for index in range(0,len(output_array)):
                    for j in range(0,1):
                        SNslope[j,index] = self.m_value

                if self.wndfiletype[i-1] == 'turb':
                    Tmax = self.Tmax_turb
                else:
                    Tmax = self.Tmax_nonturb

                allres, peaks_list, orig_data, rm_data, data_name = \
                    do_rainflow(files, output_array, SNslope, self.dir_saved_plots, Tmax, self.dT, self.rm_time, self.check_rm_time)

                a = allres[0]

                # === rainflow check === #
                if self.check_rainflow:

                    n = 0;
                    for m in output_array:

                        FAST_b = orig_data[:,m]
                        FAST_b_time = orig_data[:,0]
                        FAST_rm = rm_data[:,m]
                        FAST_rm_time = rm_data[:,0]

                        # if data_name[m] == 'RootMyb1':
                        #     f  = open('paper_plots/data_files/turb_DEM.txt', "w+")
                        #     for index in range(len(FAST_b)):
                        #         f.write(str(FAST_b[index])+'\n')
                        #     f.close()
                        #
                        #     f = open('paper_plots/data_files/turb_time.txt', "w+")
                        #     for index in range(len(FAST_b_time)):
                        #         f.write(str(FAST_b_time[index])+'\n')
                        #     f.close()



                        plt.figure()
                        plt.plot(FAST_b_time, FAST_b,'--', label='all data output')
                        plt.plot(FAST_rm_time, FAST_rm, label='used data output')

                        plt.xlabel('Time Step (s)')
                        plt.ylabel('Data')
                        # plt.title(data_name[m] + '; DEM = ' + str(a[n][0]) + ' kN*m')
                        plt.title(data_name[m] + '; DEM = ' + str(a[n][0]) + ' kN*m')

                        plt.legend()
                        # plt.savefig(self.dir_saved_plots + '/rainflow_check/' + data_name[m] + '.eps')
                        plt.savefig(self.dir_saved_plots + '/plots/rainflow_check/' + data_name[m] + '.png')
                        if data_name[m] == 'RootMyb1':
                            plt.show()
                        plt.close()

                        n += 1

                    quit()

                # peaks info
                peaks_array = dict()

                # create peaks master file
                for j in range(0,len(output_array)):
                    l = output_array[j]
                    peaks_array[str_val[l]] = []
                    peaks_array[str_val[l]].append(peaks_list[j].tolist())

                for j in range(0, len(peaks_array['RootMxb1'])):
                    peaks_master_x['root'].append(peaks_array['RootMxb1'][j])
                for j in range(0, len(peaks_array['RootMyb1'])):
                    peaks_master_y['root'].append(peaks_array['RootMyb1'][j])
                for j in range(0,len(peaks_array)):
                    for l in range(0, self.NBlGages[k]):
                        for m in range(0, len(peaks_array['Spn{0}MLxb1'.format(str(l+1))])):
                            peaks_master_x['bld_gage_' + str(self.BldGagNd[k][l])].append(peaks_array['Spn{0}MLxb1'.format(str(l+1))][m])
                        for m in range(0, len(peaks_array['Spn{0}MLyb1'.format(str(l + 1))])):
                            peaks_master_y['bld_gage_' + str(self.BldGagNd[k][l])].append(peaks_array['Spn{0}MLyb1'.format(str(l+1))][m])

                # addition of turbulent safety factor
                if self.wndfiletype[spec_caseid] == 'turb':
                    a = a*self.turb_sf

                # create xRoot, xDEM, yRoot, and yDEM
                xRoot = a[0][0]

                xDEM = []
                for l in range(0,self.NBlGages[k]):
                    xDEM.append(a[l+1][0])

                yRoot = a[1+self.NBlGages[k]][0]
                yDEM = []
                for l in range(0,self.NBlGages[k]):
                    yDEM.append(a[l+2+self.NBlGages[k]][0])

                for j in range(0,self.NBlGages[k]):
                    if j == 0:
                        Edg_param = 'RootMxb1'
                        Flp_param = 'RootMyb1'
                    else:
                        Edg_param = 'Spn{0}MLxb1'.format(j)
                        Flp_param = 'Spn{0}MLyb1'.format(j)

                    Edg_max_val = abs(max(resultsdict[Edg_param]))
                    Edg_min_val = abs(min(resultsdict[Edg_param]))

                    Flp_max_val = abs(max(resultsdict[Flp_param]))
                    Flp_min_val = abs(min(resultsdict[Flp_param]))

                    if j == 0:
                        Edg_max_array[i-1][0] = max(Edg_max_val, Edg_min_val)
                        Flp_max_array[i-1][0] = max(Flp_max_val, Flp_min_val)
                    else:
                        Edg_max_array[i-1][self.BldGagNd[k][j]] = max(Edg_max_val, Edg_min_val)
                        Flp_max_array[i-1][self.BldGagNd[k][j]] = max(Flp_max_val, Flp_min_val)

                # take max at each position
                Edg_max[0] = max(Edg_max[0], Edg_max_array[i - 1][0])
                Flp_max[0] = max(Flp_max[0], Flp_max_array[i - 1][0])
                for j in range(1, len(self.BldGagNd[k])+1):

                    for l in range(0,len(tot_BldGagNd)):
                        if tot_BldGagNd[l] == self.BldGagNd[k][j - 1]:
                            max_it = l

                    Edg_max[max_it] = max(Edg_max[max_it], Edg_max_array[i - 1][self.BldGagNd[k][j - 1]])
                    Flp_max[max_it] = max(Flp_max[max_it], Flp_max_array[i - 1][self.BldGagNd[k][j - 1]])

                # Add DEMs to master arrays
                DEMx_master_array[i-1][0] = xRoot
                DEMy_master_array[i-1][0] = yRoot

                DEMx_master_array[i-1][self.BldGagNd[k]] = xDEM
                DEMy_master_array[i-1][self.BldGagNd[k]] = yDEM

                # take max at each position
                DEMx_max[0] = max(DEMx_max[0], DEMx_master_array[i - 1][0])
                DEMy_max[0] = max(DEMy_max[0], DEMy_master_array[i - 1][0])

                for j in range(1, len(self.BldGagNd[k]) + 1):

                    for l in range(0,len(tot_BldGagNd)):
                        if tot_BldGagNd[l] == self.BldGagNd[k][j - 1]:
                            max_it = l+1

                    DEMx_max[max_it] = max(DEMx_max[max_it], DEMx_master_array[i-1][self.BldGagNd[k][j-1]])
                    DEMy_max[max_it] = max(DEMy_max[max_it], DEMy_master_array[i-1][self.BldGagNd[k][j-1]])

                if self.Run_Once:

                    # save root values

                    # xRoot file
                    xRoot_file = FAST_wnd_directory + '/' + 'xRoot.txt'
                    file_xroot = open(xRoot_file, "w")

                    # yRoot file
                    yRoot_file = FAST_wnd_directory + '/' + 'yRoot.txt'
                    file_yroot = open(yRoot_file, "w")

                    # write to xDEM file
                    file_xroot.write(str(xRoot) + '\n')
                    file_xroot.close()

                    # write to yDEM file
                    file_yroot.write(str(yRoot) + '\n')
                    file_yroot.close()

                    # save xDEM, yDEM

                    # xDEM file
                    xDEM_file = FAST_wnd_directory + '/' + 'xDEM_' + str(self.BldGagNd[k][0]) + '.txt'
                    file_x = open(xDEM_file, "w")

                    # yDEM file
                    yDEM_file = FAST_wnd_directory + '/' + 'yDEM_' + str(self.BldGagNd[k][0]) + '.txt'
                    file_y = open(yDEM_file, "w")

                    for j in range(0,len(xDEM)):

                        # write to xDEM file
                        file_x.write(str(xDEM[j]) + '\n')

                        # write to yDEM file
                        file_y.write(str(yDEM[j]) + '\n')

                    file_x.close()
                    file_y.close()

                # === turbulent extreme moment extrapolation === #

                if self.wndfiletype[spec_caseid] == 'turb':

                    from scipy.stats import norm

                    for j_index in range(0, 2):  # for both x,y bending moments

                        if j_index == 1:
                            peaks_master = peaks_master_x
                            data_type = 'x'
                        else:
                            peaks_master = peaks_master_y
                            data_type = 'y'

                        for i_index in range(0, total_num_bl_gages+1): # +1 for root bending moment

                            if i_index == 0:
                                data_name = 'root'
                            else:
                                data_name = 'bld_gage_' + str(tot_BldGagNd[i_index-1])
                            root_peaks = peaks_master[data_name]

                            # get data
                            rp_list = []
                            for m_index in range(0, len(root_peaks)):
                                for j in range(0, len(root_peaks[m_index])):
                                    rp_list.append(root_peaks[m_index][j])

                            # multi modal distribution
                            if data_type == 'x':

                                # gaussian distribution
                                if self.eme_fit == 'gaussian':
                                    # get fit
                                    data = rp_list

                                    if len(data) == 0:
                                        pass
                                    else:

                                        data_min = min(data)
                                        data_max = max(data)

                                        data1_subset = []
                                        data2_subset = []

                                        for k_index in range(0, len(data)):
                                            if abs(data[k_index] - data_min) < abs(data[k_index] - data_max):
                                                data1_subset.append(data[k_index])
                                            else:
                                                data2_subset.append(data[k_index])

                                        for l in range(2):

                                            if l == 0:
                                                data = data1_subset
                                            else:
                                                data = data2_subset

                                            plt.figure()
                                            # Fit a normal distribution to the data:
                                            mu, std = norm.fit(data)

                                            # Plot the histogram.
                                            plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')

                                            # Plot the PDF.
                                            xmin, xmax = plt.xlim()
                                            x = np.linspace(xmin, xmax, 100)
                                            p = norm.pdf(x, mu, std)
                                            plt.plot(x, p, 'k', linewidth=2)
                                            plt.title(data_name + data_type + ' Turbulent Peaks, Gaussian Fit, data subset: ' + str(l+1))
                                            plt.ylabel('Normalized Frequency')
                                            plt.xlabel('Load Bins (kN*m)')

                                            # add extrapolated, extreme moment
                                            spec_sd = 3.7*10.0**(-8.0)

                                            extreme_mom = max(abs(mu + std*norm.ppf(spec_sd)), abs(mu + std*norm.ppf(1.0-spec_sd)))

                                            # checks max, since we're doing it for both data subsets
                                            peaks_wnd_x[str(i)][data_name] = max(extreme_mom, peaks_wnd_x[str(i)][data_name])


                                            # show plot, quit routine
                                            if self.check_peaks:
                                                plt.savefig(
                                                    self.dir_saved_plots + '/plots/hist_' + str(data_name) + str(data_type) + '_' + str(l) + '.png')
                                                plt.show()
                                                # quit()
                                            plt.close()

                                else:
                                    raise Exception('Distribution specified is not implemented.')

                            # one distribution
                            elif data_type == 'y':

                                if self.eme_fit == 'gaussian':
                                    # get fit
                                    data = rp_list

                                    if len(data) == 0:
                                        pass
                                    else:

                                        plt.figure()
                                        # Fit a normal distribution to the data:
                                        mu, std = norm.fit(data)

                                        # Plot the histogram.
                                        plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')

                                        # Plot the PDF.
                                        xmin, xmax = plt.xlim()
                                        x = np.linspace(xmin, xmax, 100)
                                        p = norm.pdf(x, mu, std)
                                        plt.plot(x, p, 'k', linewidth=2)
                                        plt.title(data_name + data_type + ' Turbulent Peaks, Gaussian Fit')
                                        # plt.title('Turbulent Peaks, Gaussian Fit example')
                                        plt.ylabel('Normalized Frequency')
                                        plt.xlabel('Load Bins (kN*m)')

                                        # add extrapolated, extreme moment
                                        spec_sd = 3.7 * 10.0 ** (-8.0)

                                        extreme_mom = max(abs(mu + std * norm.ppf(spec_sd)),
                                                          abs(mu + std * norm.ppf(1.0 - spec_sd)))

                                        peaks_wnd_y[str(i)][data_name] = extreme_mom

                                        # show plot, quit routine
                                        if self.check_peaks:


                                            if data_name + data_type == 'rooty': #'RootMyb1':
                                                f  = open('paper_plots/data_files/peak_x.txt', "w+")
                                                for index in range(len(x)):
                                                    f.write(str(x[index])+'\n')
                                                f.close()

                                                f = open('paper_plots/data_files/peak_p.txt', "w+")
                                                for index in range(len(p)):
                                                    f.write(str(p[index])+'\n')
                                                f.close()

                                                f = open('paper_plots/data_files/peak_data.txt', "w+")
                                                for index in range(len(data)):
                                                    f.write(str(data[index])+'\n')
                                                f.close()

                                                quit()

                                            plt.savefig(
                                                self.dir_saved_plots + '/plots/hist_' + str(data_name) + str(data_type) + '.png')
                                            plt.show()
                                            # quit()
                                        plt.close()

                                else:
                                    raise Exception('Distribution specified is not implemented.')

        # === determine maximum of extreme extrapolated turbulent loads === #
        for j in range(len(Edg_max)):

            cur_var_x = []
            cur_var_y = []

            for i in range(len(self.WNDfile_List)):

                if j == 0:
                    cur_var_x.append(peaks_wnd_x[str(i+1)]['root'])
                    cur_var_y.append(peaks_wnd_y[str(i+1)]['root'])
                else:
                    cur_var_x.append(peaks_wnd_x[str(i+1)]['bld_gage_' + str(tot_BldGagNd[j-1])])
                    cur_var_y.append(peaks_wnd_y[str(i+1)]['bld_gage_' + str(tot_BldGagNd[j-1])])

            if j == 0:
                peaks_max_x['root'] = max(cur_var_x)
                peaks_max_y['root'] = max(cur_var_y)
            else:
                peaks_max_x['bld_gage_' + str(tot_BldGagNd[j-1])] = max(cur_var_x)
                peaks_max_y['bld_gage_' + str(tot_BldGagNd[j-1])] = max(cur_var_y)

        if self.check_peaks:
            quit()

        # compare peaks_max_x, peaks_max_y with Edg_max, Flp_max
        for i in range(0, len(Edg_max)):
            if i == 0:
                Edg_max[i] = max(Edg_max[i],peaks_max_x['root'])
                Flp_max[i] = max(Flp_max[i],peaks_max_y['root'])
            else:
                Edg_max[i] = max(Edg_max[i], peaks_max_x['bld_gage_' + str(tot_BldGagNd[i-1])])
                Flp_max[i] = max(Flp_max[i], peaks_max_y['bld_gage_' + str(tot_BldGagNd[i-1])])


        # === structural akima spline === #

        spline_extr = params['initial_str_grid']
        spline_pos = params['rstar_damage'][np.insert(tot_BldGagNd, 0, 0.0)]

        Edg_max_spline = Akima(spline_pos, Edg_max)

        unknowns['Edg_max'] = Edg_max_spline.interp(spline_extr)[0]*10.0**3.0 # kN*m to N*m

        Flp_max_spline = Akima(spline_pos, Flp_max)
        unknowns['Flp_max'] = Flp_max_spline.interp(spline_extr)[0]*10.0**3.0 # kN*m to N*m

        # === DEM akima spline === #

        spline_extr = params['rstar_damage']
        spline_pos = params['rstar_damage'][np.insert(tot_BldGagNd,0,0.0)]

        DEMx_spline = Akima(spline_pos,DEMx_max)
        unknowns['DEMx'] = DEMx_spline.interp(spline_extr)[0]

        DEMy_spline = Akima(spline_pos,DEMy_max)
        unknowns['DEMy'] = DEMy_spline.interp(spline_extr)[0]

        # kN*m to N*m
        unknowns['DEMx'] *= 1000.0
        unknowns['DEMy'] *= 1000.0


        if self.check_sgp_spline:

            # plot splines
            spline_plot = np.linspace(0,1,200)
            DEMx_spline_plot = Akima(spline_pos, DEMx_max)
            DEMx_plot = DEMx_spline_plot.interp(spline_plot)[0]

            DEMy_spline = Akima(spline_pos, DEMy_max)
            DEMy_plot = DEMy_spline.interp(spline_plot)[0]

            # DEMx
            plt.figure()
            plt.plot(spline_pos, DEMx_max, '*', label='points')
            plt.plot(spline_plot, DEMx_plot, label='spline')

            plt.xlabel('Unit Radius of Blade')
            plt.ylabel('DEMx (kN*m)')
            plt.title('DEMx spline - ' + str(total_num_bl_gages) + ' strain gages')
            plt.legend()

            plt.savefig(self.dir_saved_plots + "/plots/DEM_plots/DEMx_nsg" + str(total_num_bl_gages) + ".png")
            plt.close()

            # DEMy
            # plt.figure()
            # plt.plot(spline_pos, DEMy_max, '*', label='points')
            # # plt.plot(spline_extr, unknowns['DEMy'], label='spline')
            # plt.plot(spline_plot, DEMy_plot, label='spline')
            #
            # plt.xlabel('Unit Radius of Blade')
            # plt.ylabel('DEMy (N*m)')
            # plt.title('DEMy spline - ' + str(total_num_bl_gages) + ' strain gages')
            # plt.legend()
            #
            # plt.savefig(self.dir_saved_plots + "/plots/DEM_plots/DEMy_nsg" + str(total_num_bl_gages) + ".png")
            # print('saved at ')
            # print(self.dir_saved_plots + "/plots/DEM_plots/DEMy_nsg" + str(total_num_bl_gages) + ".png")
            # print(unknowns['DEMy'])
            # print(spline_pos)
            #
            # plt.show()
            # plt.close()


            # DEMx spline comparison

            spline_pos17 = [0. ,    0.022 , 0.067,  0.111 , 0.167 , 0.233 , 0.3  ,  0.367 , 0.433 , 0.5  ,  0.567,
             0.633 , 0.7 ,   0.767 , 0.833,  0.889,  0.933 , 0.978]
            DEMx_max17 = [1.15730078e+04  , 1.03837989e+04 ,  8.31564948e+03  , 6.59095802e+03,
             5.20400486e+03 ,  4.11302713e+03 ,  3.26303203e+03  , 2.56273712e+03,
             1.97383316e+03  , 1.47258899e+03  , 1.05731342e+03 ,  7.24606846e+02,
             4.65608939e+02  , 2.69168838e+02 ,  1.31655484e+02,   5.00200946e+01,
             1.43189321e+01 ,  9.83791912e-01]

            spline_plot = np.linspace(0,1,200)
            DEMx_spline_plot17 = Akima(spline_pos17, DEMx_max17)
            DEMx_plot17 = DEMx_spline_plot17.interp(spline_plot)[0]

            plt.figure()

            plt.plot(spline_pos17, DEMx_max17, 'g*')
            plt.plot(spline_plot, DEMx_plot17, 'g', label='Spline using 17 points')

            plt.plot(spline_pos, DEMx_max, 'b*')
            plt.plot(spline_plot, DEMx_plot, 'b--', label='Spline using 7 points')

            plt.xlabel('Unit Radius of Blade')
            plt.ylabel('DEMx (kN*m)')
            plt.title('DEMx spline comparison')
            plt.legend()

            plt.savefig(self.dir_saved_plots + "/plots/DEM_plots/DEMx_comparison.png")

            plt.show()
            plt.close()

            quit()

        # === tip deflection constraint === #

        max_tip_def_array = np.zeros([len(self.caseids), 1])

        for i in range(0, len(self.caseids)):
            resultsdict = params[self.caseids[i]]

            maxdeflection = abs(max(resultsdict['OoPDefl1']))
            mindeflection = abs(min(resultsdict['OoPDefl1']))

            max_tip_def_array[i - 1] = max(maxdeflection, mindeflection)

        unknowns['max_tip_def'] = max(max_tip_def_array)


class Blade_Damage(Group):
    def __init__(self, FASTinfo, naero=17, nstr=38):
        super(Blade_Damage, self).__init__()

        # params to calculate DEMs
        self.add('chord_sub', IndepVarComp('chord_sub', np.zeros(4),units='m'), promotes=['*'])
        self.add('theta_sub', IndepVarComp('theta_sub', np.zeros(4), units='deg'), promotes=['*'])
        self.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])

        self.add('sparT', IndepVarComp('sparT', val=np.zeros(5), units='m', desc='spar cap thickness parameters'),
                 promotes=['*'])
        self.add('teT', IndepVarComp('teT', val=np.zeros(5), units='m', desc='trailing-edge thickness parameters'),
                 promotes=['*'])

        # params to calculate training points for surrogate model
        self.add('turbine_class', IndepVarComp('turbine_class', val=Enum('I', 'II', 'III'), desc='IEC turbine class', pass_by_obj=True), promotes=['*'])
        self.add('turbulence_class', IndepVarComp('turbulence_class', val=Enum('B', 'A', 'C'), desc='IEC turbulence class', pass_by_obj=True), promotes=['*'])
        self.add('turbulence_intensity', IndepVarComp('turbulence_intensity', val=0.14, desc='IEC turbulence class intensity', pass_by_obj=False), promotes=['*'])
        self.add('af_idx', IndepVarComp('af_idx', val=np.zeros(naero), pass_by_obj=True), promotes=['*'])
        self.add('airfoil_types', IndepVarComp('airfoil_types', val=np.zeros(8), pass_by_obj=True), promotes=['*'])
        self.add('nBlades', IndepVarComp('nBlades', 3, pass_by_obj=True), promotes=['*'])
        self.add('bladeLength', IndepVarComp('bladeLength', 0.0, units='m'), promotes=['*'])
        self.add('initial_aero_grid', IndepVarComp('initial_aero_grid', np.zeros(naero)), promotes=['*'])
        self.add('initial_str_grid', IndepVarComp('initial_str_grid', np.zeros(nstr)), promotes=['*'])
        self.add('rstar_damage', IndepVarComp('rstar_damage', val=np.zeros(naero+1), desc='nondimensional radial locations of damage equivalent moments'), promotes=['*'])


        # === use surrogate model of FAST outputs === #
        if FASTinfo['Use_FAST_sm']:

            # create fit - can check to see if files already created either here or in component
            pkl_file_name = FASTinfo['opt_dir'] + '/' + 'sm_x' + '_' + FASTinfo['approximation_model'] + '.pkl'
            if not os.path.isfile(pkl_file_name):

                self.add('FAST_sm_fit', calc_FAST_sm_fit(FASTinfo, naero, nstr))

            # use fit
            self.add('use_FAST_sm_fit', use_FAST_surr_model(FASTinfo, naero, nstr),
                     promotes=['DEMx_sm', 'DEMy_sm', 'Flp_sm', 'Edg_sm', 'def_sm'])

        # === use FAST === #
        if FASTinfo['use_FAST']:

            WND_File_List = FASTinfo['wnd_list']

            # create WNDfile case ids
            caseids = FASTinfo['caseids']

            self.add('FASTconfig', CreateFASTConfig(naero, nstr, FASTinfo, WND_File_List, caseids),
                     promotes=['cfg_master'])

            from FST7_aeroelasticsolver import FST7Workflow, FST7AeroElasticSolver

            self.add('ParallelFASTCases', FST7AeroElasticSolver(caseids, FASTinfo['Tmax_turb'],
                                                                FASTinfo['Tmax_nonturb'], FASTinfo['rm_time'],
                                                                FASTinfo['wnd_type_list'], FASTinfo['dT'],
                                                                FASTinfo['output_list']))

            self.connect('cfg_master', 'ParallelFASTCases.cfg_master')

            self.add('FASTConstraints', CreateFASTConstraints(naero, nstr, FASTinfo, WND_File_List, caseids),
                     promotes=['DEMx', 'DEMy', 'max_tip_def', 'Edg_max', 'Flp_max'])

            for i in range(len(caseids)):
                self.connect('ParallelFASTCases.' + caseids[i], 'FASTConstraints.' + caseids[i])

            self.connect('cfg_master', 'FASTConstraints.cfg_master')

            if FASTinfo['calc_surr_model']:
                self.add('calc_FAST_sm_training_points', Calculate_FAST_sm_training_points(FASTinfo, naero, nstr))

        # === necessary connections ==== #

        if FASTinfo['use_FAST']:

            # FAST config
            self.connect('chord_sub', 'FASTconfig.chord_sub')
            self.connect('theta_sub', 'FASTconfig.theta_sub')

            # FAST Constraints
            self.connect('initial_aero_grid', 'FASTConstraints.initial_aero_grid')
            self.connect('initial_str_grid', 'FASTConstraints.initial_str_grid')
            self.connect('rstar_damage', 'FASTConstraints.rstar_damage')

        # train FAST surrogate model points
        if FASTinfo['train_sm']:
            # FAST outputs
            self.connect('Flp_max', 'calc_FAST_sm_training_points.Flp_max')
            self.connect('Edg_max', 'calc_FAST_sm_training_points.Edg_max')
            self.connect('DEMx', 'calc_FAST_sm_training_points.DEMx')
            self.connect('DEMy', 'calc_FAST_sm_training_points.DEMy')
            self.connect('max_tip_def', 'calc_FAST_sm_training_points.max_tip_def')

            # design variables
            self.connect('r_max_chord', 'calc_FAST_sm_training_points.r_max_chord')
            self.connect('chord_sub', 'calc_FAST_sm_training_points.chord_sub')
            self.connect('theta_sub', 'calc_FAST_sm_training_points.theta_sub')
            self.connect('sparT', 'calc_FAST_sm_training_points.sparT')
            self.connect('teT', 'calc_FAST_sm_training_points.teT')

            # calculate chord / blade length
            self.connect('bladeLength', 'calc_FAST_sm_training_points.bladeLength')
            self.connect('nBlades', 'calc_FAST_sm_training_points.nBlades')

            # other surrogate model inputs
            self.connect('turbine_class', 'calc_FAST_sm_training_points.turbine_class')
            self.connect('turbulence_class', 'calc_FAST_sm_training_points.turbulence_class')
            self.connect('af_idx', 'calc_FAST_sm_training_points.af_idx')
            self.connect('airfoil_types', 'calc_FAST_sm_training_points.airfoil_types')

        # FAST surrogate model
        if FASTinfo['Use_FAST_sm']:

            # design variables
            self.connect('r_max_chord', 'use_FAST_sm_fit.r_max_chord')
            self.connect('chord_sub', 'use_FAST_sm_fit.chord_sub')
            self.connect('theta_sub', 'use_FAST_sm_fit.theta_sub')
            self.connect('sparT', 'use_FAST_sm_fit.sparT')
            self.connect('teT', 'use_FAST_sm_fit.teT')

            # calculate chord / blade length
            self.connect('bladeLength', 'use_FAST_sm_fit.bladeLength')
            self.connect('nBlades', 'use_FAST_sm_fit.nBlades')

            # other surrogate model inputs
            self.connect('turbine_class', 'use_FAST_sm_fit.turbine_class')
            self.connect('turbulence_class', 'use_FAST_sm_fit.turbulence_class')
            self.connect('af_idx', 'use_FAST_sm_fit.af_idx')
            self.connect('airfoil_types', 'use_FAST_sm_fit.airfoil_types')

            self.connect('bladeLength', 'FAST_sm_fit.bladeLength')
            self.connect('nBlades', 'FAST_sm_fit.nBlades')

            self.connect('r_max_chord', 'FAST_sm_fit.r_max_chord')
            self.connect('chord_sub', 'FAST_sm_fit.chord_sub')
            self.connect('theta_sub', 'FAST_sm_fit.theta_sub')
            self.connect('sparT', 'FAST_sm_fit.sparT')
            self.connect('teT', 'FAST_sm_fit.teT')

if __name__=="__main__":

	pass
