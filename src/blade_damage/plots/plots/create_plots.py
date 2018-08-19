import matplotlib.pyplot as plt
import numpy as np
from akima import Akima
from scipy.stats import norm
import re
import matplotlib

from math import sin, cos, radians

def switch_values(x):

    for i in range(len(x)):
        if x[i] == 0:
            x[i] = 1
        elif x[i] == 1:
            x[i] = 0
        else:
            raise Exception('All values must be 0 or 1.')

    return x

def plots_for_presentation():

    # ========================================================================================== #


    fig = plt.gcf()
    ax = fig.gca()

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.title('Overall RMS for test: ' + file_dir_name)
    # ax.set_title('Overall RMS')

    ax.set_xticks([1,2,3,4,5,6])
    ax.set_xticklabels(['LS', 'Quad', 'RBF', 'Kriging', 'KPLS', 'KPLSK'],fontsize=12)
    ax.set_xlim([0,7])
    ax.set_xlabel('Surrogate Fit Type',fontsize=14)

    ax.set_ylim([0,8])
    ax.set_ylabel('RMS (%)',fontsize=14)

    ax.plot([1,2,3,4,5,6], [6.83, 3.52, 3.66, 3.08, 3.05, 3.06], 'bo')

    plt.savefig('rms.png')
    # plt.show()
    plt.close()

    # ========================================================================================== #
    # Forces @ root in edgewise direction

    f = open('data_files/RootFxc1.txt', "r")
    lines = f.readlines()

    RootFxc1 = []
    for i in range(len(lines)):
        RootFxc1.append(float(lines[i]))
    f.close()


    plt.figure(figsize=[8,6])

    plt.plot(RootFxc1, 'b')

    plt.ylabel('Edgewise Root Load (kN)', fontsize=14)
    plt.xlabel('Simulation Step', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('RootFxc1_turb.png')
    # plt.show()
    plt.close()

    # Moment @ root in flapwise direction

    f = open('data_files/RootMyb1.txt', "r")
    lines = f.readlines()

    RootMyb1 = []
    for i in range(len(lines)):
        RootMyb1.append(float(lines[i]))
    f.close()

    RootMyb1_short = []
    for i in range(len(RootMyb1)/1):
        RootMyb1_short.append(RootMyb1[i*1])


    # plt.figure(figsize=[8,6])
    #
    # ax = plt.subplot(111)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    #
    # plt.plot(RootMyb1_short)
    #
    # plt.ylabel('Flapwise Root Moment (kN)', fontsize=14)
    # plt.xlabel('Simulation Step', fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    #
    # plt.savefig('RootMyb1_turb.png')
    # # plt.show()
    # plt.close()


    # ========================================================================================== #
    # out of plane deflection for blade 1

    f = open('data_files/OoPDefl1.txt', "r")
    lines = f.readlines()

    OoPDefl1 = []
    for i in range(len(lines)):
        OoPDefl1.append(float(lines[i]))
    f.close()


    plt.figure(figsize=[8,6])

    plt.plot(OoPDefl1, 'b')

    plt.ylabel('Out-of-Plane Tip Deflection (m)', fontsize=14)
    plt.xlabel('Simulation Step', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('OoPDefl1_turb.png')
    # plt.show()
    plt.close()

    # quit()


    # ========================================================================================== #
    f = open('data_files/nonturb_time.txt', "r")
    lines = f.readlines()

    nonturb_time = []
    for i in range(len(lines)):
        nonturb_time.append(float(lines[i]))
    f.close()

    f = open('data_files/nonturb_DEM.txt', "r")
    lines = f.readlines()

    nonturb_DEM = []
    for i in range(len(lines)):
        nonturb_DEM.append(float(lines[i]))
    f.close()


    BM_peak = [nonturb_DEM[0]]
    BM_peak_time = [nonturb_time[0]]
    pos = 1
    neg = 0
    for i in range(1, len(nonturb_DEM)):

        # check that previous value is greater and pos is true
        if pos == 1 and nonturb_DEM[i] >= nonturb_DEM[i-1]:
            pass
        elif neg == 1 and nonturb_DEM[i] <= nonturb_DEM[i-1]:
            pass
        else:
            [pos, neg] = switch_values([pos, neg])

            BM_peak.append(nonturb_DEM[i])
            BM_peak_time.append(nonturb_time[i])

    BM_peak.append(nonturb_DEM[-1])
    BM_peak_time.append(nonturb_time[-1])

    plt.figure(figsize=[8,6])

    plt.plot(nonturb_time, nonturb_DEM, 'b')

    # plt.plot(BM_peak_time, BM_peak, 'go')

    # offset = 2
    # plt.plot(BM_peak_time[0+offset:2+offset], BM_peak[0+offset:2+offset], 'go')

    plt.ylabel('Flapwise Root Moment (kN*m)', fontsize=14)
    plt.xlabel('Simulation Time (s)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # plt.savefig('RootMyb1_nonturb_pres_4.png')
    plt.savefig('RootMyb1_nonturb_pres.png')
    # plt.show()
    plt.close()

    # ========================================================================================== #
    # twist distribution
    theta_sub = np.array([13.2783, 7.46036, 2.89317, -0.0878099])

    theta_pos = np.linspace(0.0, 1.0, 4)

    theta_spline = Akima(theta_pos, theta_sub)

    spline_points = np.linspace(0.0, 1.0, 200)
    spline_theta = theta_spline.interp(spline_points)[0]

    plt.figure(figsize=[8,6])

    plt.plot(spline_points, spline_theta, 'b', label='spline')
    plt.plot(theta_pos, theta_sub, 'go', label='control points', markersize=10)


    plt.xlabel('Unit Radius of Blade', fontsize=14)
    plt.ylabel('Twist (m)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)

    plt.savefig('theta_pres.png')

    # plt.show()
    plt.close()

    # ========================================================================================== #
    # spar cap thickness
    sparT = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])

    sparT_pos = np.linspace(0.0, 1.0, 5)

    sparT_spline = Akima(sparT_pos, sparT)

    spline_points = np.linspace(0.0, 1.0, 200)
    spline_sparT = sparT_spline.interp(spline_points)[0]

    plt.figure(figsize=[8,6])

    plt.plot(spline_points, spline_sparT, 'b', label='spline')
    plt.plot(sparT_pos, sparT, 'go', label='control points', markersize=10)


    plt.xlabel('Unit Radius of Blade', fontsize=14)
    plt.ylabel('Spar Cap Thickness (m)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)

    plt.savefig('sparT_pres.png')

    # plt.show()
    plt.close()

    # ========================================================================================== #
    # trailing edge thickness
    teT = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])

    teT_pos = np.linspace(0.0, 1.0, 5)

    teT_spline = Akima(teT_pos, teT)

    spline_points = np.linspace(0.0, 1.0, 200)
    spline_teT = teT_spline.interp(spline_points)[0]

    plt.figure(figsize=[8,6])

    plt.plot(spline_points, spline_teT, 'b', label='spline')
    plt.plot(teT_pos, teT, 'go', label='control points', markersize=10)


    plt.xlabel('Unit Radius of Blade', fontsize=14)
    plt.ylabel('Trailing Edge Thickness (m)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)

    plt.savefig('teT_pres.png')

    # plt.show()
    plt.close()

    # ========================================================================================== #
    # chord distribution

    chord_sub = np.array([3.2612, 4.5709, 3.3178, 1.4621])
    chord_pos = np.linspace(0.11111057, 1.0, 4)

    chord_spline = Akima(chord_pos, chord_sub)

    spline_points = np.linspace(0.11111057, 1.0, 200)
    spline_chord = chord_spline.interp(spline_points)[0]

    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(spline_points, spline_chord, 'b', label='spline')
    plt.plot(chord_pos, chord_sub, 'go', label='control points', markersize=10)


    plt.xlabel('Unit Radius of Blade', fontsize=14)
    plt.ylabel('Chord Length (m)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12, frameon=False)

    plt.savefig('chord_pres.png')

    # plt.show()
    plt.close()

    # ========================================================================================== #

    pass

def plots_for_paper():

    # ========================================================================================== #
    # nonturbulent wind file plots

    f = open('data_files/EWSV+12.0.wnd')
    lines = f.readlines()
    data = []
    for i in range(0, len(lines)):
        data.append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lines[i].strip('\n')))
    f.close()

    # !   Time  HorSpd  WndDir  VerSpd  HorShr  VerShr  LnVShr  GstSpd
    Time = []
    HorSpd = []
    LnVShr = []
    for i in range(len(data)):
        Time.append(float(data[i][0]))
        HorSpd.append(float(data[i][1]))
        LnVShr.append(float(data[i][6]))


    plt.figure(figsize=[8,6])

    plt.plot(Time, HorSpd, label='Horizontal Wind Speed')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Horizontal Wind Speed (m/s)', fontsize=12)
    # plt.legend(fontsize=12)

    plt.savefig('nonturb_windspeed.png')
    # plt.show()
    plt.close()

    plt.figure(figsize=[8,6])

    plt.plot(Time, LnVShr, 'green')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Vertical Wind Shear Intensity', fontsize=12)
    # plt.legend(fontsize=12)

    plt.savefig('nonturb_shr.png')
    # plt.show()
    plt.close()

    # ========================================================================================== #
    # turbulent wind file plots

    f = open('data_files/dlc_1ETM_seed1_mws11.hh')
    lines = f.readlines()
    data = []
    for i in range(8, len(lines)):
        data.append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lines[i].strip('\n')))
    f.close()

    # !   Time  HorSpd  WndDir  VerSpd  HorShr  VerShr  LnVShr  GstSpd
    Time = []
    HorSpd = []
    WndDir = []
    VerSpd =[]
    for i in range(len(data)):
        Time.append(float(data[i][0]))
        HorSpd.append(float(data[i][1]))
        WndDir.append(float(data[i][2]))
        VerSpd.append(float(data[i][3]))


    plt.figure(figsize=[8,6])

    plt.plot(Time, HorSpd, label='Horizontal Wind Speed')
    plt.plot(Time, VerSpd, label='Vertical Wind Speed')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Wind Speed (m/s)', fontsize=12)
    plt.legend(fontsize=12)

    plt.savefig('turb_windspeed.png')
    # plt.show()
    plt.close()

    plt.figure(figsize=[8,6])

    plt.plot(Time, WndDir, 'green')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Wind Direction (deg)', fontsize=12)
    # plt.legend(fontsize=12)

    plt.savefig('turb_winddir.png')
    # plt.show()
    plt.close()

    # quit()

    # ========================================================================================== #

    # TUM chord distribution
    f = open('../FAST_Files/TUM_Files/TUM_335MW_chord.txt', "r")
    lines = f.readlines()
    chord_dist = np.zeros([len(lines)])
    for i in range(len(chord_dist)):
        chord_dist[i] = float(lines[i])
    f.close()

    f = open('../FAST_Files/TUM_Files/TUM_335MW_chord_pos.txt', "r")
    lines = f.readlines()
    chord_pos = np.zeros([len(lines)])
    for i in range(len(chord_pos)):
        chord_pos[i] = float(lines[i])
    f.close()
    chord_pos = (chord_pos - chord_pos[0]) / (chord_pos[-1] - chord_pos[0])

    chord_pos_plot = chord_pos
    chord_dist_plot = chord_dist

    rotor_points = np.linspace(0, 1, 4)
    points17 = np.linspace(0, 1, 170)

    chord_spline = Akima(chord_pos, chord_dist)
    chord_sub = chord_spline.interp(rotor_points)[0] / 1000.0  # mm to m

    rotor_spline = Akima(rotor_points, chord_sub)
    rotor_chord_plot = rotor_spline.interp(points17)[0]

    plt.figure()

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(chord_pos_plot*(1-0.11111)+0.11111, chord_dist_plot / 1000.0, label='TUM 3.35 MW Chord Distribution')  # mm to m
    plt.plot(points17*(1-0.11111)+0.11111, rotor_chord_plot, '--', label='RotorSE Chord Distribution Approximation')

    # plt.title('TUM 3.35MW Chord Approximation')
    plt.xlabel('Blade Fraction')
    plt.ylabel('Chord Length (m)')
    plt.legend()

    plt.savefig('tum_chord.png')

    # plt.show()
    plt.close()

    # ========================================================================================== #

    # TUM twist distribution
    f = open('../FAST_Files/TUM_Files/TUM_335MW_twist.txt', "r")
    lines = f.readlines()
    twist_dist = np.zeros([len(lines)])
    for i in range(len(twist_dist)):
        twist_dist[i] = float(lines[i])
    f.close()

    f = open('../FAST_Files/TUM_Files/TUM_335MW_twist_pos.txt', "r")
    lines = f.readlines()
    twist_pos = np.zeros([len(lines)])
    for i in range(len(twist_pos)):
        twist_pos[i] = float(lines[i])
    f.close()
    twist_pos = (twist_pos - twist_pos[0]) / (twist_pos[-1] - twist_pos[0])

    twist_pos_plot = twist_pos
    twist_dist_plot = twist_dist

    rotor_points = np.linspace(0, 1, 4)
    points17 = np.linspace(0, 1, 170)

    twist_spline = Akima(twist_pos, twist_dist)
    theta_sub = twist_spline.interp(rotor_points)[0]

    rotor_spline = Akima(rotor_points, theta_sub)
    rotor_twist_plot = rotor_spline.interp(points17)[0]

    plt.figure()

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(twist_pos_plot*(1-0.166667)+0.166667, twist_dist_plot, label='TUM 3.35 MW Twist Distribution')
    plt.plot(points17*(1-0.166667)+0.166667, rotor_twist_plot, '--', label='RotorSE Twist Distribution Approximation')

    # plt.title('TUM 3.35MW Twist Approximation')
    plt.xlabel('Blade Fraction')
    plt.ylabel('Blade Twist (deg)')
    plt.legend()

    plt.savefig('tum_twist.png')

    # plt.show()
    plt.close()

    # quit()

    # ========================================================================================== #
    # wind turbine illustration plot

    fig = plt.gcf()
    ax = fig.gca()

    D = np.array([25.0, 35.0, 49.5, 64.0])*2.0
    hubD = np.array([1.25, 1.75, 2.475, 3.2])*2.0
    towerH = np.array([58.67, 82.39, 116.73, 151.07])

    spacing = 0.9
    X = np.array([1.0*D[3], 2.0*D[3], 3.25*D[3], 4.75*D[3]])*spacing

    colors = ['grey', 'green', 'blue', 'green']
    label = ['Unused', 'Trained', 'Predicted', 'Trained']
    # colors = ['blue', 'green', 'green', 'green']
    # label = ['Predicted', 'Trained', 'Trained', 'Trained']

    for i in range(4):

        # D1 = 126.239
        D1 = D[i]
        R1 = D1/2.

        c1 = R1/35.
        # H1 = 109.067
        H1 = towerH[i]
        # r = 126.4/2.
        r = D1/2.0
        # x1 = 1.5*r
        x1 = X[i]
        # x2 = x1 + 126.4 * spacing

        # hub circle

        hub1 = plt.Circle((x1,H1), 3*c1, color=colors[i], fill=False, linewidth=2)
        ax.add_artist(hub1)

        # blade circle

        circle1 = plt.Circle((x1,H1), R1, color=colors[i], linestyle='--', fill=False, linewidth=2)
        ax.add_artist(circle1)

        # tower

        d1 = np.array([6.3, 6.3, 3.87])

        px1 = np.array([x1-d1[0]/2,x1-d1[1]/2,x1-d1[2]/2,x1+d1[2]/2,x1+d1[1]/2,x1+d1[0]/2,x1-d1[0]/2])
        py1 = np.array([0,H1/2,H1-3.*c1,H1-3.*c1,H1/2,0,0])

        if i < 3:
        # if i < 2:
            ax.plot(px1, py1, colors[i], linewidth=2, label=label[i])
        else:
            ax.plot(px1, py1, colors[i], linewidth=2)

        # blades

        bladeX = np.array([3.,7.,10.,15.,20.,25.,30.,35.,30.,25.,20.,15.,10.,5.,3.,3.])
        bladeY = np.array([0.,0.,0.8,1.5,1.7,1.9,2.1,2.3,2.4,2.4,2.4,2.4,2.4,2.4,2.4,0.])-1.5
        angle1 = np.random.rand(1)*60.-55.

        blade1X = bladeX*cos(radians(angle1))-bladeY*sin(radians(angle1))
        blade1Y = bladeX*sin(radians(angle1))+bladeY*cos(radians(angle1))
        blade2X = bladeX*cos(radians(angle1+120.))-bladeY*sin(radians(angle1+120.))
        blade2Y = bladeX*sin(radians(angle1+120.))+bladeY*cos(radians(angle1+120.))
        blade3X = bladeX*cos(radians(angle1+240.))-bladeY*sin(radians(angle1+240.))
        blade3Y = bladeX*sin(radians(angle1+240.))+bladeY*cos(radians(angle1+240.))
        ax.plot(blade1X*c1+x1, blade1Y*c1+H1, colors[i], linewidth=2)
        ax.plot(blade2X*c1+x1, blade2Y*c1+H1, colors[i], linewidth=2)
        ax.plot(blade3X*c1+x1, blade3Y*c1+H1, colors[i], linewidth=2)

    # axes, other options

    plt.axes().set_aspect('equal')
    ax.set_yticks([50, 100, 150, 200, 250])
    ax.set_ylim([0, 250])


    ax.set_xticks(X)
    ax.set_yticks([])
    ax.set_xticklabels(['0.75 MW', '1.5 MW', '3.0 MW', '5.0 MW'],fontsize=10)
    # ax.set_xticklabels(['', '', '', ''],fontsize=10)
    ax.set_yticklabels([],fontsize=10)
    ax.set_xlim([0.25*D[3], 5.25*D[3]])

    # ax.set_ylabel('Height (m)', fontsize=12)
    # ax.set_xlabel('WindPACT reference turbines', fontsize=12)
    plt.legend()

    plt.savefig('3mw_prediction_front.eps')
    # plt.savefig('075mw_prediction.png')

    # plt.show()
    plt.close()

    # quit()

    # ========================================================================================== #
    # chord, twist domain plot

    old_var_list = [[1.3, 5.3], [1.3, 5.3], [1.3, 5.3],
                    [1.3, 5.3], [-10.0, 30.0], [-10.0, 30.0], [-10.0, 30.0], [-10.0, 30.0]]

    new_var_list =[[2.2612000000000001, 4.2612000000000005], [3.5709, 5.3],
                   [2.3178000000000001, 4.3178000000000001], [1.3, 2.4621],
     [3.2782999999999998, 23.278300000000002], [-2.5396400000000003, 17.460360000000001],
     [-7.1068300000000004, 12.89317], [-10.0, 9.9121901000000001]]

    # chord_sub domain plot
    plt.figure()
    j = 1

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(4):

        plt.plot([j, j], old_var_list[i], 'bx')
        plt.plot([j, j], new_var_list[i], '--rx')
        j += 1

    plt.xticks(np.linspace(1, j - 1, j - 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('chord variable index', fontsize=16)
    plt.ylabel('Var. Domain (m)', fontsize=16)
    # plt.title('chord domain, restriction: ' + str(FASTinfo['range_frac'] * 100.0) + '%')

    plt.savefig('chord_sub_domain.png')
    # plt.show()

    plt.close()

    # theta_sub domain plot
    plt.figure()

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    j = 1


    for i in range(4,8):
        plt.plot([j, j], old_var_list[i], 'bx')
        plt.plot([j, j], new_var_list[i], '--rx')
        j += 1


    plt.xticks(np.linspace(1, j - 1, j - 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('twist variable index', fontsize=16)
    plt.ylabel('Var. Domain (deg)', fontsize=16)
    # plt.title('twist domain, restriction: ' + str(FASTinfo['range_frac'] * 100.0) + '%')

    plt.savefig('theta_sub_domain.png')
    # plt.show()

    plt.close()

    # ========================================================================================== #
    # cv plot

    f = open('data_files/pointfile500.txt')
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lines[i].strip('\n'))

    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(200):
        if i < 100:
            if i == 0:
                plt.plot(float(lines[i][0]) * 4.0 + 1.3, float(lines[i][1]) * 4.0 + 1.3, 'bo', label = 'Training Points')
            else:
                plt.plot(float(lines[i][0]) * 4.0 + 1.3, float(lines[i][1]) * 4.0 + 1.3, 'bo')

        else:
            if i == 100:
                plt.plot(float(lines[i][0])*4.0+1.3, float(lines[i][1])*4.0+1.3, 'ro', label = 'Cross Validation Points')
            else:
                plt.plot(float(lines[i][0])*4.0+1.3, float(lines[i][1])*4.0+1.3, 'ro')

    plt.xlabel('1st Point in Chord Distribution (m)', fontsize=16-2)
    plt.ylabel('2nd Point in Chord Distribution (m)', fontsize=16-2)
    plt.xticks(np.linspace(1.3,5.3,5),fontsize=14-2)
    plt.yticks(np.linspace(1.3,5.3,5),fontsize=14-2)
    plt.legend(bbox_to_anchor=(.35, 1.15), loc=1, borderaxespad=0., fontsize=14-2)
    # plt.legend(fontsize=14-2, )

    plt.xlim([1.25, 5.35])
    plt.ylim([1.25, 5.35])

    plt.savefig('cv_example.png')
    # plt.show()

    plt.close()

    # ========================================================================================== #
    # kfold plot

    f = open('data_files/pointfile500.txt')
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lines[i].strip('\n'))

    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(100):
        if i < 20:
            if i == 0:
                plt.plot(float(lines[i][0]) * 4.0 + 1.3, float(lines[i][1]) * 4.0 + 1.3, 'ro', label = '1st k-fold group')
            else:
                plt.plot(float(lines[i][0]) * 4.0 + 1.3, float(lines[i][1]) * 4.0 + 1.3, 'ro')

        else:
            plt.plot(float(lines[i][0])*4.0+1.3, float(lines[i][1])*4.0+1.3, 'bo')

    plt.xlabel('1st Point in Chord Distribution (m)', fontsize=16-2)
    plt.ylabel('2nd Point in Chord Distribution (m)', fontsize=16-2)
    plt.xticks(np.linspace(1.3,5.3,5),fontsize=14-2)
    plt.yticks(np.linspace(1.3,5.3,5),fontsize=14-2)
    # plt.legend(bbox_to_anchor=(.35, 1.15), loc=1, borderaxespad=0., fontsize=14-2)
    plt.legend(fontsize=14-2)

    plt.xlim([1.25, 5.35])
    plt.ylim([1.25, 5.35])

    plt.savefig('kfold_example.png')
    # plt.show()

    plt.close()

    # ========================================================================================== #
    # lhs plot

    f = open('data_files/pointfile500.txt')
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lines[i].strip('\n'))

    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(100):
        plt.plot(float(lines[i][0])*4.0+1.3, float(lines[i][1])*4.0+1.3, 'bo')

    plt.xlabel('1st Point in Chord Distribution (m)', fontsize=16-2)
    plt.ylabel('2nd Point in Chord Distribution (m)', fontsize=16-2)
    plt.xticks(np.linspace(1.3,5.3,5), fontsize=14-2)
    plt.yticks(np.linspace(1.3,5.3,5), fontsize=14-2)

    plt.xlim([1.25, 5.35])
    plt.ylim([1.25, 5.35])

    plt.savefig('lhs.png')
    # plt.show()

    plt.close()


    # ========================================================================================== #
    # linear plot

    one_dim = np.linspace(1.3,5.3, 10)

    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(10):
        for j in range(10):
            plt.plot(one_dim[i], one_dim[j], 'bo')

    plt.xlabel('1st Point in Chord Distribution (m)', fontsize=16-2)
    plt.ylabel('2nd Point in Chord Distribution (m)', fontsize=16-2)
    plt.xticks(np.linspace(1.3,5.3,5), fontsize=14-2)
    plt.yticks(np.linspace(1.3,5.3,5), fontsize=14-2)

    plt.xlim([1.25, 5.35])
    plt.ylim([1.25, 5.35])

    plt.savefig('linear.png')
    # plt.show()

    plt.close()

    # ========================================================================================== #
    # DEMx at NTM, 5-15 m/s, NREL 5MW

    mws5 = [1.06809244e+04,   9.77844276e+03,   7.85636713e+03,   6.25407249e+03,
     4.96444652e+03, 3.94730316e+03,   3.14959231e+03,   2.48922145e+03,
     1.92055688e+03,   1.43220331e+03,   1.02681594e+03,   7.01675432e+02,
     4.50662671e+02,   2.61061533e+02,   1.28195860e+02,   4.92434627e+01,
     1.42476788e+01,   9.66286025e-01]
    mws7 = [1.08695760e+04,   1.00427717e+04,   8.08179230e+03,   6.44963643e+03,
     5.13362321e+03,   4.09417489e+03,   3.27702775e+03,   2.59659247e+03,
     1.99961703e+03,   1.48551051e+03,   1.06218221e+03,   7.23680320e+02,
     4.67286526e+02,   2.73446631e+02,   1.36058124e+02,   5.38450132e+01,
     1.66350683e+01,   1.18511305e+00]
    mws9 = [1.12542168e+04,   1.04471994e+04,   8.42054280e+03,   6.73967488e+03,
     5.38929174e+03,   4.31986060e+03,   3.47479076e+03,   2.76658372e+03,
     2.13692347e+03,   1.59184894e+03,   1.14675973e+03,   7.88461862e+02,
     5.12482490e+02,   3.03598027e+02,   1.54487988e+02,   6.36142283e+01,
     2.01609217e+01,   1.45424572e+00]
    mws11 = [1.14921582e+04,   1.05217729e+04,   8.51823596e+03,   6.84651944e+03,
     5.49534894e+03,   4.41854780e+03,   3.56173523e+03,   2.84085178e+03,
     2.20194775e+03,   1.64402958e+03,   1.18095067e+03,   8.06274090e+02,
     5.19661486e+02,   3.04231213e+02,   1.53649556e+02,   6.25513295e+01,
     1.98100070e+01,   1.43025058e+00]
    mws13 = [1.13249442e+04,   1.03524948e+04,   8.36635464e+03,   6.70618792e+03,
     5.36552357e+03,   4.30656304e+03,   3.46897987e+03,   2.76510174e+03,
     2.13632342e+03,   1.58800020e+03,   1.13754958e+03,   7.75261053e+02,
     5.04025418e+02,   2.99804502e+02,   1.53046857e+02,   6.35167304e+01,
     2.02229487e+01,   1.45652202e+00]
    mws15 = [1.12096900e+04,   1.02438158e+04,   8.31385285e+03,   6.69716175e+03,
     5.38912789e+03,   4.34878973e+03,   3.52159320e+03,   2.82245451e+03,
     2.19176976e+03,   1.63620514e+03,   1.17494378e+03,   7.98537049e+02,
     5.12326797e+02,   2.96969768e+02,   1.47672228e+02,   5.69121252e+01,
     1.67817222e+01,   1.17983902e+00]

    plt.figure(figsize=(10,5))

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('DEM (kN*m)', fontsize=16)
    # plt.title('DEMx for various input .wnd files')  #: Bending Moment at Spanwise Station #1, Blade #1')

    plt.plot(mws5, label='NTM, 5 m/s')
    plt.plot(mws7, label='NTM, 7 m/s')
    plt.plot(mws9, label='NTM, 9 m/s')
    plt.plot(mws11, label='NTM, 11 m/s')
    plt.plot(mws13, label='NTM, 13 m/s')
    plt.plot(mws15, label='NTM, 15 m/s')

    plt.legend(fontsize=14, frameon=False)
    plt.xticks(np.linspace(0, 17, 18), fontsize=10)
    plt.yticks(fontsize=14)
    plt.savefig('DEMx_dif_wnd_files.png')
    # plt.show()
    plt.close()

    # ========================================================================================== #
    # maximum xDEM, yDEM

    f = open('data_files/xDEM_max.txt', "r")
    lines = f.readlines()

    xDEM_max = []
    for i in range(len(lines)):
        xDEM_max.append(float(lines[i]))
    f.close()

    f = open('data_files/yDEM_max.txt', "r")
    lines = f.readlines()

    yDEM_max = []
    for i in range(len(lines)):
        yDEM_max.append(float(lines[i]))
    f.close()

    rotor_star = [0., 0.022, 0.067, 0.111, 0.167, 0.233, 0.3, 0.367, 0.433, 0.5, 0.567,
                  0.633, 0.7, 0.767, 0.833, 0.889, 0.933, 0.978]

    plt.figure(figsize=(10,5))


    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(rotor_star, np.array(xDEM_max)*1.05/1000.0, 'blue', label='Edgewise')
    plt.plot(rotor_star, np.array(yDEM_max)*1.05/1000.0, 'green', label='Flapwise')

    # plt.title('xMaximum DEMs')
    plt.ylabel('Equivalent Fatigue Damage (kN*m)', fontsize=16)
    plt.xlabel('Blade Fraction', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=14, frameon=False)

    plt.savefig('DEM_max_plot.png')

    # plt.show()
    plt.close()

    # ========================================================================================== #
    # extreme loads plot

    f = open('data_files/peak_data.txt', "r")
    lines = f.readlines()

    data = []
    for i in range(len(lines)):
        data.append(float(lines[i]))
    f.close()

    f = open('data_files/peak_p.txt', "r")
    lines = f.readlines()

    p = []
    for i in range(len(lines)):
        p.append(float(lines[i]))
    f.close()

    f = open('data_files/peak_x.txt', "r")
    lines = f.readlines()

    x = []
    for i in range(len(lines)):
        x.append(float(lines[i]))
    f.close()

    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')

    plt.plot(x, p, 'k', linewidth=2)
    # plt.title('Turbulent Peaks, Gaussian Fit example')
    plt.ylabel('Normalized Frequency', fontsize=16-2)
    plt.xlabel('Peak Bins (kN*m)', fontsize=16-2)

    plt.xticks([-2000, 2000, 6000, 10000, 14000],fontsize=14-2)
    plt.yticks([])

    plt.xlim([-2500, 14500])
    plt.ylim([0, 1.875e-4])

    plt.savefig("rooty.png")
    # plt.savefig("rooty_bins.png")

    # plt.show()
    plt.close()

    # ========================================================================================== #
    # NREL 5 MW flapwise bending moment at root, @ DLC_1_2, 11 m/s, 1st seed


    f = open('data_files/turb_time.txt', "r")
    lines = f.readlines()

    turb_time = []
    for i in range(len(lines)):
        turb_time.append(float(lines[i]))
    f.close()

    f = open('data_files/turb_DEM.txt', "r")
    lines = f.readlines()

    turb_DEM = []
    for i in range(len(lines)):
        turb_DEM.append(float(lines[i]))
    f.close()

    plt.figure(figsize=[8,6])


    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(turb_time, turb_DEM)
    plt.ylabel('Flapwise Root Moment (kN*m)', fontsize=14)
    plt.xlabel('Simulation Time (s)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('RootMyb1_turb.png')
    # plt.show()
    plt.close()

    # ========================================================================================== #
    # NREL 5 MW flapwise bending moment at root, @ rated speed


    f = open('data_files/nonturb_time.txt', "r")
    lines = f.readlines()

    nonturb_time = []
    for i in range(len(lines)):
        nonturb_time.append(float(lines[i]))
    f.close()

    f = open('data_files/nonturb_DEM.txt', "r")
    lines = f.readlines()

    nonturb_DEM = []
    for i in range(len(lines)):
        nonturb_DEM.append(float(lines[i]))
    f.close()

    plt.figure(figsize=[8,6])


    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(nonturb_time, nonturb_DEM)
    plt.ylabel('Flapwise Root Moment (kN*m)', fontsize=14)
    plt.xlabel('Simulation Time (s)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('RootMyb1_nonturb.png')
    # plt.show()
    plt.close()


    # ========================================================================================== #
    # NREL 5MW Chord Distribution

    chord_sub = np.array([3.2612, 4.5709, 3.3178, 1.4621])
    chord_pos = np.linspace(0.11111057, 1.0, 4)

    points17 = np.linspace(0.11111057, 1.0, 17)
    list7 = [0, 1, 2, 4, 6, 8, 11, 16]
    points7 = points17[list7]
    plot_points = np.linspace(0.11111057, 1.0, 200)

    chord_spline = Akima(chord_pos, chord_sub)
    plot_spline = chord_spline.interp(plot_points)[0]
    plot17 = chord_spline.interp(points17)[0]
    plot7 = plot17[list7]

    plt.figure(figsize=[8,6])

    plt.plot(plot_points, plot_spline, '--', label='spline')
    plt.plot(points17, plot17, 'b*', label='full set')
    plt.plot(points7, plot7, 'g*', label='subset')

    plt.xlabel('Unit Radius of Blade', fontsize=14)
    plt.ylabel('Local Chord Length (m)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)

    plt.savefig('chordplot.png')

    # plt.show()
    plt.close()


    # ========================================================================================== #
    # DEMx comparison using 7 vs 17 strain gages - NREL 5MW @ rated wind speed

    spline_pos = np.array([0.   ,  0.022 , 0.111 , 0.233 , 0.367 , 0.5   , 0.7   , 0.978])
    DEMx_max = np.array([[1.03748402e+04],
                [9.31019618e+03],
                [5.91028142e+03],
                [3.68880627e+03],
                [2.29885004e+03],
                [1.32106944e+03],
                [4.17721925e+02],
                [8.81864890e-01]])

    spline_pos17 = np.array([0.  ,   0.022 , 0.067 , 0.111,  0.167 , 0.233 , 0.3  ,  0.367, 0.433 , 0.5   , 0.567,
     0.633 , 0.7 ,   0.767 , 0.833 , 0.889  ,0.933  ,0.978])
    DEMx_max17 = np.array([[1.03748402e+04],
                    [9.31019618e+03],
                    [7.45563387e+03],
                    [5.91028142e+03],
                    [4.66695798e+03],
                    [3.68880627e+03],
                    [2.92634129e+03],
                    [2.29885004e+03],
                    [1.77063283e+03],
                    [1.32106944e+03],
                    [9.48528168e+02],
                    [6.50066059e+02],
                    [4.17721925e+02],
                    [2.41527613e+02],
                    [1.18117981e+02],
                    [4.48727830e+01],
                    [1.28398106e+01],
                    [8.81864890e-01]])

    spline_plot = np.linspace(0, 1, 200)
    DEMx_spline_plot = Akima(spline_pos, DEMx_max)
    DEMx_plot = DEMx_spline_plot.interp(spline_plot)[0]

    spline_plot = np.linspace(0, 1, 200)
    DEMx_spline_plot17 = Akima(spline_pos17, DEMx_max17)
    DEMx_plot17 = DEMx_spline_plot17.interp(spline_plot)[0]

    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(spline_pos17, DEMx_max17, 'g*')
    ax.plot(spline_plot, DEMx_plot17, 'g', label='Spline using 17 points')

    ax.plot(spline_pos, DEMx_max, 'b*')
    ax.plot(spline_plot, DEMx_plot, 'b--', label='Spline using 7 points')

    ax.set_xlabel('Unit Radius of Blade', fontsize=14)
    ax.set_ylabel('Equivalent Fatigue Damage (kN*m)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title('DEMx spline comparison')
    plt.legend(fontsize=12, frameon=False)

    plt.savefig("DEMx_comparison.png")

    # plt.show()
    plt.close()

    # quit()

    # ========================================================================================== #

    avg_percent_DEM_error_x = [[ 0.03031669],
 [ 0.03231118],
 [ 0.03136817],
 [ 0.03112903],
 [ 0.03088376],
 [ 0.03066035],
 [ 0.03029632],
 [ 0.02981686],
 [ 0.02952737],
 [ 0.02924914],
 [ 0.02873618],
 [ 0.02832635],
 [ 0.02788814],
 [ 0.02767299],
 [ 0.02780215],
 [ 0.02852558],
 [ 0.03016187],
 [ 0.03348269],
 [ 0.03440994]]

    avg_percent_DEM_error_y = [[ 0.02780971],
 [ 0.02692806],
 [ 0.02697093],
 [ 0.02714859],
 [ 0.02732884],
 [ 0.0276685 ],
 [ 0.02826308],
 [ 0.0289538 ],
 [ 0.02967979],
 [ 0.03030714],
 [ 0.03061677],
 [ 0.03116511],
 [ 0.03166875],
 [ 0.03238446],
 [ 0.03333208],
 [ 0.03420149],
 [ 0.03489482],
 [ 0.03521297]]

    # DEMx plot
    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.title('DEMx k-fold check (surrogate model accuracy)')
    plt.plot(np.array(avg_percent_DEM_error_x) * 100.0, 'x', label='avg error')
    plt.xlabel('strain gage position', fontsize=16+2)
    plt.ylabel('DEMx RMS (%)', fontsize=16+2)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14+2)
    plt.yticks(fontsize=14+2)
    # plt.legend()
    plt.savefig('DEMx.png')
    # plt.show()
    plt.close()

    # DEMy plot
    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.title('DEMy k-fold check (surrogate model accuracy)')

    plt.plot(np.array(avg_percent_DEM_error_y) * 100.0, 'x', label='avg error')
    plt.xlabel('strain gage position', fontsize=16+2)
    plt.ylabel('DEMy RMS (%)', fontsize=16+2)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14+2)
    plt.yticks(fontsize=14+2)
    # plt.legend()
    plt.savefig('DEMy.png')
    # plt.show()
    plt.close()


    # ========================================================================================== #
    # cross validate with 3.0 MW

    DEMx_actual = [3.87371629e+06  , 3.51162948e+06 ,  2.96716035e+06   ,2.47457223e+06,
     2.03288631e+06  , 1.63747295e+06  , 1.29095488e+06 ,  9.96027370e+05,
     7.48782298e+05 ,  5.46820864e+05  , 3.86414239e+05 ,  2.65255258e+05,
     1.74523231e+05 ,  1.07604806e+05  , 5.93069651e+04  , 2.75128468e+04,
     9.04452297e+03 ,  1.01259292e+03]

    DEMx_sm = [3.95565057e+06 ,  3.62696809e+06 ,  3.05814893e+06 ,  2.54342154e+06,
     2.08430023e+06  , 1.67241117e+06  , 1.31488900e+06 ,  1.01092248e+06,
     7.58913095e+05 ,  5.56071728e+05 ,  3.95224478e+05  , 2.71172291e+05,
     1.77707108e+05 ,  1.09191682e+05 ,  6.01765743e+04  , 2.79280241e+04,
     9.26779552e+03 ,  1.03890128e+03]

    DEMy_actual = [3.79448773e+06 ,  3.85883635e+06  , 3.59143524e+06 ,  3.14323852e+06,
     2.71651056e+06 ,  2.29111828e+06  , 1.90904259e+06 ,  1.57178440e+06,
     1.27215891e+06 ,  1.01149992e+06  , 7.92779002e+05  , 6.06709535e+05,
     4.40324626e+05 ,  2.96885096e+05  , 1.79800156e+05  , 9.13882701e+04,
     3.30386060e+04 ,  3.90547150e+03]
    DEMy_sm = [3820492.00084065  ,3880487.92616103 , 3444724.6698005 ,  3026915.91648388,
     2629790.95426712 , 2245208.37646793 , 1891794.32959772 , 1570163.62151356,
     1280900.36207485 , 1025222.39020009 ,  802495.51307971  , 610441.14456754,
     445392.2995785  ,  304896.15666846 ,  189105.46552702  ,  99070.49800788,
     36881.7381568   ,   4337.61473042]

    # DEMx plot
    plt.figure(figsize=[8,6])
    # plt.title('DEMx initial design (surrogate model accuracy)')

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(np.array(DEMx_actual)/1000.0, '-o', label='FAST-calculated')
    plt.plot(np.array(DEMx_sm)/1000.0, '--x', label='Predicted')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('DEM (kN*m)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.savefig('DEMx_comp_3.png')
    # plt.show()
    plt.close()

    # DEMy plot
    plt.figure(figsize=[8,6])
    # plt.title('DEMx initial design (surrogate model accuracy)')

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(np.array(DEMy_actual)/1000.0, '-o', label='FAST-calculated')
    plt.plot(np.array(DEMy_sm)/1000.0, '--x', label='Predicted')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('DEM (kN*m)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.savefig('DEMy_comp_3.png')
    # plt.show()
    plt.close()


    # error plots

    # DEMx error plot
    plt.figure(figsize=[8,6])
    # plt.title('DEMx initial design (surrogate model accuracy)')

    plt.plot(abs(np.array(DEMx_actual)-np.array(DEMx_sm))/DEMx_actual[0]*100.0, 'x')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('RMS error (%)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('DEMx_comp_3_error.png')
    # plt.show()
    plt.close()

    # DEMy error plot
    plt.figure(figsize=[8,6])
    # plt.title('DEMx initial design (surrogate model accuracy)')

    plt.plot(abs(np.array(DEMy_actual)-np.array(DEMy_sm))/DEMy_actual[0]*100.0, 'x')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('RMS error (%)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('DEMy_comp_3_error.png')
    # plt.show()
    plt.close()


    # ========================================================================================== #
    # cross validate with WP 0.75 MW

    DEMx_actual = [3.09739299e+05 ,  3.09578600e+05 ,  2.64083148e+05 ,  2.22361462e+05,
     1.84752980e+05 ,  1.47935607e+05 ,  1.15783900e+05 ,  8.85479209e+04,
     6.59394565e+04 ,  4.80436376e+04 ,  3.47887116e+04  , 2.47326367e+04,
     1.68142595e+04 ,  1.07663301e+04 ,  6.17190131e+03 ,  2.97866087e+03,
     1.01896719e+03 ,  1.16805848e+02]

    DEMx_sm = [3.14715693e+05 ,  3.14507869e+05 ,  2.90437620e+05  , 2.47231911e+05,
     2.00298570e+05 ,  1.59548883e+05  , 1.25063045e+05 ,  9.64054558e+04,
     7.30672511e+04 ,  5.38763279e+04  , 3.81773817e+04  , 2.53023508e+04,
     1.56011016e+04  , 8.75696093e+03  , 4.16971826e+03 ,  1.55201191e+03,
     3.66926185e+02  , 4.19159470e+01]

    DEMy_actual = [ 363675.0105108 ,  384437.3371893 ,  343429.0488675 ,
        303173.9335341 ,  263956.8463839 ,  225762.3899055 ,
        189733.4184483 ,  156502.6427439 ,  127869.6135393 ,
        102624.8496084 ,   80706.7697058 ,   62201.64643092,
         45619.60491324,   31153.54922694,   19083.52824318,
          9877.11425505,    3707.85224848,     450.60625317]

    DEMy_sm = [339658.68360703 , 364427.81410405 , 325619.06369935 , 288629.92692659,
     253250.41881599 , 216973.23013148 , 183627.74129907 , 153097.77804593,
     125610.99175433 , 101447.59322582 ,  79932.57425244 ,  60552.75365609,
     43647.24892832  , 29422.56839565 ,  17870.64782469  ,  9126.34015663,
     3222.42874383   ,  396.73826183]

    # DEMx plot
    plt.figure(figsize=[8,6])
    # plt.title('DEMx initial design (surrogate model accuracy)')

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(np.array(DEMx_actual)/1000.0, '-o', label='FAST-calculated')
    plt.plot(np.array(DEMx_sm)/1000.0, '--x', label='Predicted')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('DEM (kN*m)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.savefig('DEMx_comp_075.png')
    # plt.show()
    plt.close()

    # DEMy plot
    plt.figure(figsize=[8,6])
    # plt.title('DEMx initial design (surrogate model accuracy)')

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(np.array(DEMy_actual)/1000.0, '-o', label='FAST-calculated')
    plt.plot(np.array(DEMy_sm)/1000.0, '--x', label='Predicted')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('DEM (kN*m)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.savefig('DEMy_comp_075.png')
    # plt.show()
    plt.close()

    # error plots

    # DEMx error plot
    plt.figure(figsize=[8,6])
    plt.plot(abs(np.array(DEMx_actual)-np.array(DEMx_sm))/DEMx_actual[0]*100.0, 'x')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('RMS error (%)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('DEMx_comp_075_error.png')
    # plt.show()
    plt.close()

    # DEMy error plot
    plt.figure(figsize=[8,6])
    plt.plot(abs(np.array(DEMy_actual)-np.array(DEMy_sm))/DEMy_actual[0]*100.0, 'x')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('RMS error (%)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('DEMy_comp_075_error.png')
    # plt.show()
    plt.close()

    # ========================================================================================== #
    # cross validate with NREL 5MW

    DEMx_actual = [1.14921582e+07,   1.05217729e+07,   8.51823596e+06,   6.84651944e+06,
     5.49534894e+06,   4.41854780e+06,   3.56173523e+06,   2.84085178e+06,
     2.20194775e+06,   1.64402958e+06,   1.18095067e+06,   8.06274090e+05,
     5.19661486e+05,   3.04231213e+05,   1.54487988e+05,   6.36142283e+04,
     2.02229487e+04,   1.45652202e+03]
    DEMx_sm = [1.20909023e+07,   1.09674603e+07,   9.22647301e+06,   7.65187555e+06,
     6.24837543e+06,   5.00874922e+06,   3.93223904e+06,   3.01548788e+06,
     2.25368470e+06,   1.63822583e+06,   1.15089953e+06,   7.75824488e+05,
     4.96258192e+05,   2.95354647e+05,   1.56605840e+05,   6.90993066e+04,
     2.15624383e+04,   2.37475028e+03]
    DEMy_actual = [1.07091866e+07,   1.08920195e+07,   9.52801891e+06,   8.34258185e+06,
     7.23995050e+06,   6.21473062e+06,   5.26481252e+06,   4.42358376e+06,
     3.65949598e+06,   2.95410127e+06,   2.30938342e+06,   1.73271748e+06,
     1.23200911e+06,   8.10506343e+05,   4.77732047e+05,   2.31778079e+05,
     7.59819321e+04,   5.58903850e+03]
    DEMy_sm = [1.00596319e+07,   1.04967088e+07,   9.26709108e+06,   8.09338689e+06,
     6.98421597e+06,   5.89711914e+06,   4.90757163e+06,   4.01605929e+06,
     3.22124535e+06,   2.52771784e+06,   1.92915419e+06,   1.42468317e+06,
     1.00907026e+06,  6.68990016e+05,   3.99653464e+05,   2.00418304e+05,
     7.08703926e+04,   8.20068523e+03]

    # DEMx plot
    plt.figure(figsize=[8,6])
    # plt.title('DEMx initial design (surrogate model accuracy)')

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(np.array(DEMx_actual)/1000.0*1.05, '-o', label='FAST-calculated')
    plt.plot(np.array(DEMx_sm)/1000.0*1.05, '--x', label='Predicted')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('DEM (kN*m)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.savefig('DEMx_comp_NREL_5MW.png')
    # plt.show()
    plt.close()

    # DEMy plot
    plt.figure(figsize=[8,6])
    # plt.title('DEMx initial design (surrogate model accuracy)')

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(np.array(DEMy_actual)/1000.0, '-o', label='FAST-calculated')
    plt.plot(np.array(DEMy_sm)/1000.0, '--x', label='Predicted')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('DEM (kN*m)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.savefig('DEMy_comp_NREL_5MW.png')
    # plt.show()
    plt.close()

    # DEMx error plot
    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(abs(np.array(DEMx_actual)-np.array(DEMx_sm))/DEMx_actual[0]*100.0, 'x')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('RMS error (%)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('DEMx_comp_NREL_5MW_error.png')
    # plt.show()
    plt.close()

    # DEMy error plot
    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(abs(np.array(DEMy_actual)-np.array(DEMy_sm))/DEMy_actual[0]*100.0, 'x')
    plt.xlabel('strain gage position', fontsize=16)
    plt.ylabel('RMS error (%)', fontsize=16)
    plt.xticks([0,3,6,9,12,15,17], fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('DEMy_comp_NREL_5MW_error.png')
    # plt.show()
    plt.close()

    # ========================================================================================== #

    num_pts = np.array([200, 400, 600, 800, 1000, 1200, 2000])
    data = np.array([0.072383, 0.061087, 0.045095, 0.036810, 0.031382, 0.031026, 0.030505])

    plt.figure(figsize=(10,5))

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(num_pts, data*100.0, '--*', label='Kriging')
    plt.xlabel('Number of training points for each wind turbine', fontsize=16)
    plt.ylabel('RMS Error (%)', fontsize=16)
    # plt.title('Overall RMS for surrogate model (Kriging): ')
    # plt.legend()
    plt.xticks(num_pts, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0,8])

    plt.savefig('num_comparison.png')
    # plt.show()
    plt.close()


    # ========================================================================================== #
    # fixed COE plots
    # for next 3 plots
    run_num = 5
    include_opt = 1
    include_no_fatigue = 1

    fixed_dict = dict()

    fixed_dict['Run1'] = dict()
    fixed_dict['Run2'] = dict()
    fixed_dict['Run3'] = dict()
    fixed_dict['Run4'] = dict()
    fixed_dict['Run5'] = dict()

    fixed_dict['Run1']['COE'] = 0.0617949048822
    fixed_dict['Run1']['r_max_chord'] = 1.0/3.0
    fixed_dict['Run1']['chord_sub'] = [2.76401188  ,3.78729007 , 2.97729151,  1.76964213]
    fixed_dict['Run1']['theta_sub'] = [12.04210597  , 4.62729358   ,2.76688384 ,  0.76758987]
    fixed_dict['Run1']['sparT'] = [0.09808879 , 0.07792606  ,0.05143962 , 0.03378576 , 0.00501371]
    fixed_dict['Run1']['teT'] = [0.07490683 , 0.02445826 , 0.02189605,  0.00877243 , 0.00677891]

    fixed_dict['Run2']['COE'] = 0.0621681139212
    fixed_dict['Run2']['r_max_chord'] = 1.0/3.0
    fixed_dict['Run2']['chord_sub'] = [2.80456707 , 3.78688318 , 2.9766075  , 1.62198578]
    fixed_dict['Run2']['theta_sub'] = [11.84145614  , 4.59715927 ,  2.76688384 ,  0.07603945]
    fixed_dict['Run2']['sparT'] = [0.089593  ,  0.0798049 ,  0.05030549,  0.03439933,  0.00525689]
    fixed_dict['Run2']['teT'] = [0.07384034, 0.02445879 , 0.02156773, 0.00896017,  0.00638067]

    fixed_dict['Run3']['COE'] = 0.0623285471825
    fixed_dict['Run3']['r_max_chord'] =1.0/3.0
    fixed_dict['Run3']['chord_sub'] = [2.8027549  , 3.79028559 , 3.02621062 , 1.61858908]
    fixed_dict['Run3']['theta_sub'] = [11.45970995 ,  4.72288642 ,  3.0377737,    0.76336198]
    fixed_dict['Run3']['sparT ']= [0.0936929,   0.07985158,  0.0502527   ,0.03529714,  0.00544712]
    fixed_dict['Run3']['teT'] = [0.07075389 , 0.02455161 , 0.02380444  ,0.0086941  , 0.00502962]

    fixed_dict['Run4']['COE'] = 0.0623744774342
    fixed_dict['Run4']['r_max_chord'] = 1.0/3.0
    fixed_dict['Run4']['chord_sub'] = [2.7780686  , 3.78733812 ,3.07730918 , 1.60343337]
    fixed_dict['Run4']['theta_sub'] = [11.95346389  , 4.6305542   , 3.29092767,   0.71585925]
    fixed_dict['Run4']['sparT'] = [0.09116796,  0.08196152 , 0.04855227 , 0.03701417 , 0.00513763]
    fixed_dict['Run4']['teT'] = [0.07506529 , 0.02805818  ,0.02380449 , 0.00898228 , 0.01999627]

    fixed_dict['Run5']['COE'] = 0.0623794121129
    fixed_dict['Run5']['r_max_chord'] = 1.0/3.0
    fixed_dict['Run5']['chord_sub'] = [2.7780686  , 3.78733812 ,3.07730918 , 1.60343337]
    fixed_dict['Run5']['theta_sub'] = [11.95346389  , 4.6305542   , 3.29092767,   0.71585925]
    fixed_dict['Run5']['sparT'] = [0.09116796,  0.08196152 , 0.04855227 , 0.03701417 , 0.00513763]
    fixed_dict['Run5']['teT'] = [0.07506529 , 0.02805818  ,0.02380449 , 0.00898228 , 0.01999627]

    fixed_dict['Run_opt'] = dict()
    fixed_dict['Run_opt']['COE'] = 0.062397300998

    fixed_dict['Run_no_fatigue'] = dict()
    fixed_dict['Run_no_fatigue']['COE'] = 0.0598933658009
    fixed_dict['Run_no_fatigue']['r_max_chord'] = 1.0/3.0
    fixed_dict['Run_no_fatigue']['chord_sub'] = [1.69028083,  3.91922485,  3.04194506,  1.81409819]
    fixed_dict['Run_no_fatigue']['theta_sub'] = [14.72653147,   5.83767972,   2.77269064,   0.42626158]

    # COE plot
    plt.figure(figsize=[8,6])

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(1,run_num+1):
        plt.plot(i, fixed_dict['Run' + str(i)]['COE']*100.0, 'bo') # $ to cents

    if include_opt == 1:
        plt.plot([1,5], [fixed_dict['Run_opt']['COE']*100.0, fixed_dict['Run_opt']['COE']*100.0],
                 'b--', label='In-the-loop Optimization' )

    if include_no_fatigue == 1:
        plt.plot([1,5], [fixed_dict['Run_no_fatigue']['COE']*100.0, fixed_dict['Run_no_fatigue']['COE']*100.0],
                 'r--', label='Optimization with Static Loading')

    # plt.legend(fontsize=14-2)
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel('Iteration Number', fontsize=16-2)
    plt.ylabel('COE (cents/kWh)', fontsize=16-2)
    plt.xticks(fontsize=14-2)
    plt.yticks(fontsize=14-2)
    plt.xlim([0.8, 5.2])
    plt.legend(fontsize=14-2, frameon=False)
    # plt.ylim([0.06175*100.0, 0.06245*100.0])
    # plt.title('Fixed-DEM Optimization Iterations')
    if include_no_fatigue == 1:
        plt.savefig('fixedplot_COE_opt_no_fatigue.png')
    elif include_opt == 1:
        plt.savefig('fixedplot_COE_opt.png')
    else:
        plt.savefig('fixedplot_COE_' + str(run_num) + '.png')

    # plt.show()
    plt.close()
    # quit()

    # ========================================================================================== #
    # chord plot for iterative method
    chord_pos = np.linspace(0.11111057, 1.0, 4)

    spline_plot = np.linspace(0.11111057,1,100)

    chord1 = Akima(chord_pos, fixed_dict['Run1']['chord_sub'])
    fixed_dict['chord1_spline'] = chord1.interp(spline_plot)[0]

    chord2 = Akima(chord_pos, fixed_dict['Run2']['chord_sub'])
    fixed_dict['chord2_spline'] = chord2.interp(spline_plot)[0]

    chord3 = Akima(chord_pos, fixed_dict['Run3']['chord_sub'])
    fixed_dict['chord3_spline'] = chord3.interp(spline_plot)[0]

    chord4 = Akima(chord_pos, fixed_dict['Run4']['chord_sub'])
    fixed_dict['chord4_spline'] = chord4.interp(spline_plot)[0]

    chord5 = Akima(chord_pos, fixed_dict['Run5']['chord_sub'])
    fixed_dict['chord5_spline'] = chord5.interp(spline_plot)[0]

    labels = ['1st Iteration', '2nd Iteration', '3rd Iteration', '4th Iteration', '5th Iteration']

    plt.figure(figsize=[8,6])


    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(1, run_num+1):
        plt.plot(spline_plot, fixed_dict['chord' + str(i) + '_spline'], label=labels[i-1])

    if include_no_fatigue == 1:
        chord_no_fatigue = Akima(np.linspace(0.11111057, 1, 4), fixed_dict['Run_no_fatigue']['chord_sub'])
        fixed_dict['chord_no_fatigue_spline'] = chord_no_fatigue.interp(spline_plot)[0]

        plt.plot(spline_plot, fixed_dict['chord_no_fatigue_spline'], '--', label='Static Loading')

    plt.xlabel('Blade Fraction', fontsize=16)
    plt.ylabel('Chord Length (m)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([0.05, 1.05])
    # plt.ylim([1.4, 3.9])
    # plt.title('Fixed-DEM Chord Distributions')
    plt.legend(fontsize=14, frameon=False)

    if include_no_fatigue == 1:
        plt.savefig('fixedplot_chord_no_fatigue.png')
    else:
        plt.savefig('fixedplot_chord_' + str(run_num) + '.png')

    # plt.show()
    plt.close()

    # ========================================================================================== #
    # twist plot for iterative method
    spline_plot = np.linspace(0.16666667,1,100)

    twist1 = Akima(np.linspace(0.16666667,1,4), fixed_dict['Run1']['theta_sub'])
    fixed_dict['theta1_spline'] = twist1.interp(spline_plot)[0]

    twist2 = Akima(np.linspace(0.16666667,1,4), fixed_dict['Run2']['theta_sub'])
    fixed_dict['theta2_spline'] = twist2.interp(spline_plot)[0]

    twist3 = Akima(np.linspace(0.16666667,1,4), fixed_dict['Run3']['theta_sub'])
    fixed_dict['theta3_spline'] = twist3.interp(spline_plot)[0]

    twist4 = Akima(np.linspace(0.16666667,1,4), fixed_dict['Run4']['theta_sub'])
    fixed_dict['theta4_spline'] = twist4.interp(spline_plot)[0]

    twist5 = Akima(np.linspace(0.16666667,1,4), fixed_dict['Run5']['theta_sub'])
    fixed_dict['theta5_spline'] = twist5.interp(spline_plot)[0]

    plt.figure(figsize=[8,6])


    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(1, run_num+1):
        plt.plot(spline_plot, fixed_dict['theta' + str(i) + '_spline'], label=labels[i-1])

    if include_no_fatigue == 1:

        twist_no_fatigue = Akima(np.linspace(0.16666667, 1, 4), fixed_dict['Run_no_fatigue']['theta_sub'])
        fixed_dict['theta_no_fatigue_spline'] = twist_no_fatigue.interp(spline_plot)[0]

        plt.plot(spline_plot, fixed_dict['theta_no_fatigue_spline'], '--', label='Static Loading')

    plt.xlabel('Blade Fraction', fontsize=16)
    plt.ylabel('Blade Twist (deg)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([0.1, 1.1])
    # plt.ylim([-0.5, 12.75])
    # plt.title('Fixed-DEM Twist Distributions')
    plt.legend(fontsize=14, frameon=False)

    if include_no_fatigue == 1:
        plt.savefig('fixedplot_theta_no_fatigue.png')
    else:
        plt.savefig('fixedplot_theta_' + str(run_num) + '.png')

    # plt.show()
    plt.close()

    # ========================================================================================== #
    # Optimal Chord/Twist using surrogate model

    Run_sur = dict()

    Run_sur['COE'] = 0.0623644092354
    Run_sur['r_max_chord'] = 1.0/3.0
    Run_sur['chord_sub'] = [2.781224  , 3.78766434 ,3.07731128 , 1.60349901]
    Run_sur['theta_sub'] = [12.008324  , 4.6455834   , 3.289953,   0.7160923]

    # chord plot for surr_model
    spline_plot = np.linspace(0.11111057, 1, 100)

    chord_pos = np.linspace(0.11111057,1,4)
    chord = Akima(chord_pos, fixed_dict['Run4']['chord_sub'])
    chord_spline = chord.interp(spline_plot)[0]

    plt.figure()

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(spline_plot, chord_spline)

    plt.xlabel('Blade Fraction', fontsize=16)
    plt.ylabel('Chord Length (m)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.title('Chord Distribution using Surrogate Model')
    plt.savefig('surrmodel_chord.png')
    # plt.show()
    plt.close()

    # twist plot for surr_model
    spline_plot = np.linspace(0.166667, 1, 100)

    twist_pos = np.linspace(0.166667,1,4)
    twist = Akima(twist_pos, fixed_dict['Run4']['theta_sub'])
    twist_spline = twist.interp(spline_plot)[0]

    plt.figure()

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(spline_plot, twist_spline)

    plt.xlabel('Blade Fraction', fontsize=16)
    plt.ylabel('Twist (deg)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.title('Twist Distribution using Surrogate Model')
    plt.savefig('surrmodel_twist.png')
    # plt.show()
    plt.close()


    # ========================================================================================== #


    pass

if __name__ == "__main__":

    plots_for_paper()
    plots_for_presentation()