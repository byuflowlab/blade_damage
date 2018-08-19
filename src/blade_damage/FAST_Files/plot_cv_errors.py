import numpy as np
import matplotlib.pyplot as plt

def compare_fits():

    # file_dir_name = "sc_test_1_var"
    # file_dir_name = "sc_test_all_var"

    file_dir_name = "test_sgp_3"

    file_dir = "Opt_Files/" + file_dir_name

    num_pts = 100

    file_template = "error_"
    fit_types = ["second_order_poly", "least_squares", "kriging", "KPLS", "KPLSK"]
    # fit_types = ["second_order_poly", "least_squares", "KPLS", "KPLSK"]

    data = np.zeros([len(fit_types), 2])

    for i in range(len(fit_types)):

        file_name = file_template + fit_types[i] + "_" + str(num_pts) + ".txt"

        f = open(file_dir + "/" + file_name)
        lines = f.readlines()

        for j in range(len(lines)):
            data[i,j] = float(lines[j])

    # print(data)

    plt.figure()

    max_data = []

    for i in range(len(fit_types)):
        plt.plot(i, ((data[i,0]**2.0+data[i,1]**2.0)/2.0)**0.5*100.0/5.0, 'o', label=fit_types[i])
        max_data.append(((data[i,0]**2.0+data[i,1]**2.0)/2.0)**0.5*100.0/5.0)

    print(min(max_data))

    plt.ylabel('RMS (%)')
    # plt.title('Overall RMS for test: ' + file_dir_name)
    plt.title('Overall RMS')

    plt.ylim([0,6])

    plt.legend()
    plt.savefig('/Users/bingersoll/Desktop/rms.png')
    # plt.savefig(file_dir_name + '.png')
    plt.show()
    plt.close()

def compare_num_training_points():

    file_dir_names = ['test_100', 'test_500', 'test_1000', 'test_2000']
    num_pts = [100, 500, 1000, 2000]

    data = np.zeros([len(file_dir_names), 2])

    for i in range(len(file_dir_names)):

        file_dir = "Opt_Files/" + file_dir_names[i]

        file_name = 'error_second_order_poly_' + str(num_pts[i]) + '.txt'

        file_path = file_dir + '/' + file_name


        f = open(file_path)
        lines = f.readlines()

        for j in range(len(lines)):
            data[i,0] = float(lines[0])
            data[i,1] = float(lines[1])

    plt.figure()

    plt.plot(num_pts, data[:,0]*100.0/2.0, '--*', label='Turbulence Class A')
    plt.plot(num_pts, data[:,0]*100.0/1.8, '--x', label='Turbulence Class B')
    plt.xlabel('Number of training points')
    plt.ylabel('RMS (%)')
    plt.title('Overall RMS for surrogate model (Kriging): ')
    plt.legend()

    plt.ylim([0,8])

    plt.savefig('/Users/bingersoll/Desktop' + '/num_comparison.png')
    plt.show()
    plt.close()

if __name__ == '__main__':

    # compare_fits()

    compare_num_training_points()