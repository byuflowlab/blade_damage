import os, re

def combine_results(src_dirs, dest_dir, src_num, turb, turb_class, af):

    sm_var_dir = 'sm_var_dir_' + turb + '_' + turb_class + '_' + af

    # make new destination directory
    if not os.path.isdir('Opt_Files/' + dest_dir):
        os.mkdir('Opt_Files/' + dest_dir)
    if not os.path.isdir('Opt_Files/' + dest_dir + '/' + sm_var_dir):
        os.mkdir('Opt_Files/' + dest_dir + '/' + sm_var_dir)

    # files
    ft = ['def', 'DEM', 'load', 'var']

    for k in range(len(ft)):
        for i in range(len(src_dirs)):

            f_src = open('Opt_Files/' + src_dirs[i] + '/' + sm_var_dir + '/sm_master_' + ft[k] + '.txt', "r")
            var_lines = f_src.readlines()
            f_src.close()

            if i == 0:
                f_dest = open('Opt_Files/' + dest_dir + '/' + sm_var_dir + '/sm_master_' + ft[k] + '.txt', "w+")
                f_dest.write(var_lines[0])
            else:
                f_dest = open('Opt_Files/' + dest_dir + '/' + sm_var_dir + '/sm_master_' + ft[k] + '.txt', "a")

            for j in range(1, min(len(var_lines),src_num[i]+1)):
                f_dest.write(var_lines[j])

            f_dest.close()

if __name__ == "__main__":

    # opt_file_srcs = ['test_075MW', 'test_15MW', 'test_3MW', 'test_5MW']
    # opt_file_srcs = ['test_15MW', 'test_5MW']
    # opt_file_srcs = ['test_15MW', 'test_3MW', 'test_5MW']
    # opt_file_srcs = ['test_15MW', 'test_3MW']
    # opt_file_srcs = ['test_15MW']
    opt_file_srcs = ['test_5MW']

    # opt_file_srcs_num = [200, 200, 200, 200]
    # opt_file_srcs_num = [50, 50, 50, 50]
    # opt_file_srcs_num = [997, 1000]
    # opt_file_srcs_num = [997, 1000, 1000]
    opt_file_srcs_num = [1000]

    # opt_file_dest = 'val_3MW'
    # opt_file_dest = 'val_075MW'
    # opt_file_dest = 'val_075MW_1'
    # opt_file_dest = 'val_075MW_2'
    opt_file_dest = 'val_NREL_5MW'

    turbulence = 'B'
    turbine_class = 'I'
    airfoils = 'af1'

    combine_results(opt_file_srcs, opt_file_dest, opt_file_srcs_num, turbulence, turbine_class, airfoils)

