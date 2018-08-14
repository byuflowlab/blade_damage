if __name__=="__main__":

            # === use surrogate model of FAST outputs === #
        if FASTinfo['Use_FAST_sm']:

            # create fit - can check to see if files already created either here or in component
            pkl_file_name = FASTinfo['opt_dir'] + '/' + 'sm_x' + '_' + FASTinfo['approximation_model'] + '.pkl'
            if not os.path.isfile(pkl_file_name):
                self.add('FAST_sm_fit', calc_FAST_sm_fit(FASTinfo, naero, nstr))

            # use fit
            self.add('use_FAST_sm_fit', use_FAST_surr_model(FASTinfo, naero, nstr), promotes=['DEMx_sm','DEMy_sm', 'Flp_sm', 'Edg_sm', 'def_sm'])

        if FASTinfo['use_FAST']:

            WND_File_List = FASTinfo['wnd_list']

            # create WNDfile case ids
            caseids = FASTinfo['caseids']

            self.add('FASTconfig', CreateFASTConfig(naero, nstr, FASTinfo, WND_File_List, caseids), promotes=['cfg_master'])

            from FST7_aeroelasticsolver import FST7Workflow, FST7AeroElasticSolver

            self.add('ParallelFASTCases', FST7AeroElasticSolver(caseids, FASTinfo['Tmax_turb'],
                FASTinfo['Tmax_nonturb'], FASTinfo['rm_time'], FASTinfo['wnd_type_list'], FASTinfo['dT'], FASTinfo['output_list']))

            self.connect('cfg_master', 'ParallelFASTCases.cfg_master')

            self.add('FASTConstraints', CreateFASTConstraints(naero, nstr, FASTinfo, WND_File_List, caseids),
                     promotes=['DEMx', 'DEMy', 'max_tip_def', 'Edg_max', 'Flp_max'])

            for i in range(len(caseids)):
                self.connect('ParallelFASTCases.' + caseids[i], 'FASTConstraints.' + caseids[i])


            self.connect('cfg_master', 'FASTConstraints.cfg_master')

            if FASTinfo['calc_surr_model']:
                self.add('calc_FAST_sm_training_points', Calculate_FAST_sm_training_points(FASTinfo, naero,nstr))
