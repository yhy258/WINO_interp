import os
import numpy as np
import torch
import scipy.stats as stats
from collections import defaultdict

def ttest(loss1, loss2, p_thres=0.01, test_name=''):
    # 1 - Shapiro-Wilk Test
    stat1, p_value1 = stats.shapiro(loss1)
    stat2, p_value2 = stats.shapiro(loss2)

    # print("Model 1 - Shapiro-Wilk Test p-value:", p_value1)
    # print("Model 2 - Shapiro-Wilk Test p-value:", p_value2)

    # 2단계: 정규성 검정 결과 해석
    if p_value1 >= 0.05 and p_value2 >= 0.05:
        
        # 3단계: Paired t-test 수행
        t_stat, p_value = stats.ttest_rel(loss1, loss2)
        print("t-statistic:", t_stat)
        print("p-value:", p_value)

        # 결과 해석
        if p_value < p_thres:
            print(f"T-TEST{test_name}: Meaningful difference")
        else:
            print(f"T-TEST{test_name}: No Meaningful difference")
    else:
        # print("The distributions do not follow the normal distribution. Conduct Wilcoxon signed-rank test.")
        
        # 4단계: Wilcoxon signed-rank test (비모수적 검정)
        w_stat, p_value = stats.wilcoxon(loss1, loss2)
        print("Wilcoxon signed-rank test statistic:", w_stat)
        print("p-value:", p_value)

        # 결과 해석
        if p_value < p_thres:
            print(f"SHAPIRO - {test_name}: Meaningful difference")
        else:
            print(f"SHAPIRO - {test_name}: No Meaningful difference")


if __name__ == "__main__":
        # Load data
    sim_names = ['single_layer','triple_layer', 'straight_waveguide', 'image_sensor']
    # sim_names = ['image_sensor']
    design_range_dict = {'single_layer': {'design_start':0.4, 'width': 0.12},'triple_layer': {'design_start':0.4, 'width': 0.12}, 'straight_waveguide': {'design_start':1.0, 'width':4.85 }, 'image_sensor': {'design_start':0.6, 'width': 3.5}}
    dfs = {}
    eval_modes = ['nmse_val_dict', 'structure_nmse_val_dict', 'near_nmse_val_dict']
    trained_wavelengths = list(range(400, 701, 20))
    for sim_name in sim_names:
        wino = torch.load(f'/data/joon/Results/WINO/three_main_results/three_main_results/retain_batch_test_wvlwise_results/{sim_name}/wino/nmse_waveprior_64dim_12layer_256_5060_auggroup4_weightsharing.pt')
        neurolight = torch.load(f'/data/joon/Results/WINO/three_main_results/three_main_results/retain_batch_test_wvlwise_results/{sim_name}/neurolight/nmse_wp_64_16layer_256_mode5060_dp01_bs32_ressetm.pt')

        all_wvls = list(range(400, 701))

        data_dict = {'WINO': wino, "NeurOLight": neurolight}

        for eval_mode in eval_modes:
            # Initialize a list to hold data for the plot
            
            result_dict = defaultdict()
            # Collecting WINO data
            for k, v in data_dict.items():
                data_nmses = []
                for wvl in trained_wavelengths:
                    nmses = v[eval_mode][str(wvl)].view(-1)
                    trained_str = "Trained wavelength" if int(wvl) in trained_wavelengths else "Untrained Wavelength"
                    for nmse in nmses:
                        data_nmses.append(nmse)
                data_nmses = torch.stack(data_nmses)
                result_dict[k] = data_nmses
            test_name = f"Trained wavelength {sim_name}_{eval_mode}"
            ttest(result_dict['WINO'], result_dict['NeurOLight'], 0.01, test_name=test_name)
            
            for k, v in data_dict.items():
                data_nmses = []
                for wvl in all_wvls:
                    if int(wvl) in trained_wavelengths:
                        continue
                    nmses = v[eval_mode][str(wvl)].view(-1)
                    for nmse in nmses:
                        data_nmses.append(nmse)
                data_nmses = torch.stack(data_nmses)
                result_dict[k] = data_nmses
            test_name = f"Untrained Wavelength {sim_name}_{eval_mode}"
            ttest(result_dict['WINO'], result_dict['NeurOLight'], 0.01, test_name=test_name)
                