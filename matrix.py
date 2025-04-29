# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

from meta_info import *

result_root_this_script = join(results_root, 'matrix')


class Matrix:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Matrix', result_root_this_script, mode=2)
        # self.dff = join(self.this_class_arr, 'Drought_timing.df')
        self.dff = join(self.this_class_arr, 'Attrib_dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        # self.rt_AI_Topt()
        # self.rt_AI_Tanomaly()
        self.rt_AI_Tanomaly_statistic()
        # self.normal_condition_AI_Tanomaly()
        # self.VPD_AI_Tanomaly()
        # self.sm_AI_Tanomaly()
        # self.sm_anomaly_AI_Tanomaly()
        # self.rt_AI_Tanomaly_line_plot()
        pass
    def copy_df(self):
        import attribution
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        dff = attribution.Attribution_Dataframe().dff
        df = T.load_df(dff)
        T.save_df(df,self.dff)
        T.df_to_excel(df, self.dff)

        pass

    def rt_AI_Topt(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        df = df[df['aridity_index']<=4]
        df = df.dropna(subset=['optimal_temp'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            # T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                continue
            T_anomaly_reshape = np.reshape(T_anomaly, (-1, 6))
            drought_year_T_anomaly = T_anomaly_reshape[1]
            # T_anomaly_mean = np.nanmean(drought_year_T_anomaly)
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_vals.append(T_anomaly_mean)
            # pix = row['pix']
            # if not pix in T_anomaly_vals_dict:
            #     T_anomaly_vals_dict[pix] = []
            # T_anomaly_vals_dict[pix].append(T_anomaly_mean)

            AI_vals.append(AI_val)
        # spatial_dict = {}
        # for pix in T_anomaly_vals_dict:
        #     vals = T_anomaly_vals_dict[pix]
        #     vals_mean = np.nanmean(vals)
        #     spatial_dict[pix] = vals_mean
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # plt.hist(T_anomaly_vals,bins=100)
        # plt.show()
        df['T_anomaly_mean'] = T_anomaly_vals
        # df = df[df['T_anomaly_mean']<=2]
        # df = df[df['T_anomaly_mean']>=-2]
        bin_range = np.linspace(0,1,15)
        T_quantile_bins =[]
        AI_quantile_bins = []
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))
            AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(len(T_quantile_bins))
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(len(T_quantile_bins))
        # plt.plot(T_quantile_bins)
        # plt.show()
        # print(len(T_quantile_bins))
        # print(len(AI_quantile_bins))
        # exit()
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(T_quantile_bins)
        # exi

        df_group_AI, _ = T.df_bin(df,'aridity_index',AI_quantile_bins)

        matrix = []
        y_label_list = []
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i,'T_anomaly_mean',T_quantile_bins)
            # print(len(df_group_T))
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['rt'].tolist()
                rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                # print(flag,rt_mean)
                flag += 1
            # print(len(matrix_i))
            print(flag)

            matrix.append(matrix_i)
        plt.imshow(matrix,cmap='RdBu',vmin=.95,vmax=1.05)
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_mean')
        plt.colorbar()
        plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        plt.yticks(range(len(y_label_list)),y_label_list)
        plt.tight_layout()
        plt.show()

        pass

    def rt_AI_Tanomaly(self):
        outdir = join(self.this_class_png,'rt_AI_Tanomaly')
        T.mkdir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        df = df[df['aridity_index']<=3]
        df = df.dropna(subset=['optimal_temp'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        NDVI_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            NDVI_anomaly = row['NDVI_progress'].tolist()
            # T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            # optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                continue
            NDVI_anomaly_reshape = np.reshape(NDVI_anomaly, (-1, 6))
            drought_year_NDVI_anomaly = NDVI_anomaly_reshape[1]
            NDVI_anomaly_mean = np.nanmean(drought_year_NDVI_anomaly)
            NDVI_anomaly_vals.append(NDVI_anomaly_mean)

            T_anomaly_reshape = np.reshape(T_anomaly, (-1, 6))
            drought_year_T_anomaly = T_anomaly_reshape[1]
            T_anomaly_mean = np.nanmean(drought_year_T_anomaly)
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_vals.append(T_anomaly_mean)
            # pix = row['pix']
            # if not pix in T_anomaly_vals_dict:
            #     T_anomaly_vals_dict[pix] = []
            # T_anomaly_vals_dict[pix].append(T_anomaly_mean)

            AI_vals.append(AI_val)
        # spatial_dict = {}
        # for pix in T_anomaly_vals_dict:
        #     vals = T_anomaly_vals_dict[pix]
        #     vals_mean = np.nanmean(vals)
        #     spatial_dict[pix] = vals_mean
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # plt.hist(T_anomaly_vals,bins=100)
        # plt.show()
        df['T_anomaly_mean'] = T_anomaly_vals
        df['NDVI_anomaly_mean'] = NDVI_anomaly_vals
        df = df[df['T_anomaly_mean']<=2]
        df = df[df['T_anomaly_mean']>=-2]
        bin_range = np.linspace(0,1,41)
        T_quantile_bins =[]
        AI_bins = np.linspace(0,2.5,26)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))
            # AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(len(T_quantile_bins))
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(len(T_quantile_bins))
        # plt.plot(T_quantile_bins)
        # plt.show()
        # print(len(T_quantile_bins))
        # print(len(AI_quantile_bins))
        # exit()
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(T_quantile_bins)
        # exi

        df_group_AI, _ = T.df_bin(df,'aridity_index',AI_bins)

        matrix = []
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i,'T_anomaly_mean',T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['NDVI_anomaly_mean'].tolist()
                rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                plt.scatter(bin_range[flag],y_label,c=rt_mean,vmin=-.6,vmax=.6,cmap='RdBu',marker='s')
                # print(flag,rt_mean)
                flag += 1
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_quantile')
        plt.colorbar()
        # plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        # plt.yticks(range(len(y_label_list))[::-1],y_label_list)
        # plt.tight_layout()
        plt.show()
        # outf = join(outdir,'matrix1.pdf')
        # plt.savefig(outf)
        # plt.close()

        pass
    def rt_AI_Tanomaly_statistic(self):
        outdir = join(self.this_class_arr, 'rt_AI_Tanomaly_statistic')
        T.mkdir(outdir,force=True)
        outf = join(outdir,'rt_AI_Tanomaly_statistic.npy')
        if os.path.exists(outf):
            arr = np.load(outf)
            arr_dryland = arr[:6,-4:]
            arr_dryland_mean = np.nanmean(arr_dryland)

            arr_humid_50_90 = arr[6:,25:36]
            arr_humid_50_90_mean = np.nanmean(arr_humid_50_90)

            arr_humid_above_90 = arr[6:, 36:]
            arr_humid_above_90_mean = np.nanmean(arr_humid_above_90)
            print(arr_humid_above_90_mean)
            # print(arr_dryland_mean)
            # plt.imshow(arr_dryland,interpolation='nearest',cmap='RdBu',vmin=-.6,vmax=.6)
            plt.imshow(arr_humid_above_90,interpolation='nearest',cmap='RdBu',vmin=-.6,vmax=.6)
            plt.colorbar()
            plt.show()
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        df = df[df['aridity_index']<=3]
        df = df.dropna(subset=['optimal_temp'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        NDVI_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            NDVI_anomaly = row['NDVI_progress'].tolist()
            # T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            # optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                continue
            NDVI_anomaly_reshape = np.reshape(NDVI_anomaly, (-1, 6))
            drought_year_NDVI_anomaly = NDVI_anomaly_reshape[1]
            NDVI_anomaly_mean = np.nanmean(drought_year_NDVI_anomaly)
            NDVI_anomaly_vals.append(NDVI_anomaly_mean)

            T_anomaly_reshape = np.reshape(T_anomaly, (-1, 6))
            drought_year_T_anomaly = T_anomaly_reshape[1]
            T_anomaly_mean = np.nanmean(drought_year_T_anomaly)
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_vals.append(T_anomaly_mean)
            # pix = row['pix']
            # if not pix in T_anomaly_vals_dict:
            #     T_anomaly_vals_dict[pix] = []
            # T_anomaly_vals_dict[pix].append(T_anomaly_mean)

            AI_vals.append(AI_val)
        df['T_anomaly_mean'] = T_anomaly_vals
        df['NDVI_anomaly_mean'] = NDVI_anomaly_vals
        df = df[df['T_anomaly_mean']<=2]
        df = df[df['T_anomaly_mean']>=-2]
        bin_range = np.linspace(0,1,41)
        T_quantile_bins =[]
        AI_bins = np.linspace(0,2.5,26)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))

        df_group_AI, _ = T.df_bin(df,'aridity_index',AI_bins)
        y_range = list(range(len(df_group_AI)))
        # print(y_range)
        # print(len(y_range))
        # exit()

        matrix = np.ones((25, 40)) * np.nan
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        flag_y = 0
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i,'T_anomaly_mean',T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['NDVI_anomaly_mean'].tolist()
                rt_mean = np.nanmean(rt)
                # rt_mean = len(rt)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                # plt.scatter(bin_range[flag],y_label,c=rt_mean,vmin=-.6,vmax=.6,cmap='RdBu',marker='s')
                matrix[flag_y,flag] = rt_mean
                # print(flag,rt_mean)
                flag += 1
            # print(len(matrix_i))
            flag_y += 1

            # matrix.append(matrix_i)
        np.save(outf, matrix)
        # matrix = np.array(matrix)
        pass

    def VPD_AI_Tanomaly(self):
        outdir = join(self.this_class_png,'VPD_AI_Tanomaly')
        T.mkdir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        # exit()
        df = df[df['aridity_index']<=3]
        # df = df.dropna(subset=['optimal_temp'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        NDVI_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            # NDVI_anomaly = row['NDVI_progress'].tolist()
            NDVI_anomaly = row['VPD-anomaly']
            # T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            # optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                continue
            NDVI_anomaly_vals.append(NDVI_anomaly)

            T_anomaly_reshape = np.reshape(T_anomaly, (-1, 6))
            drought_year_T_anomaly = T_anomaly_reshape[1]
            T_anomaly_mean = np.nanmean(drought_year_T_anomaly)
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_vals.append(T_anomaly_mean)
            # pix = row['pix']
            # if not pix in T_anomaly_vals_dict:
            #     T_anomaly_vals_dict[pix] = []
            # T_anomaly_vals_dict[pix].append(T_anomaly_mean)

            AI_vals.append(AI_val)
        # spatial_dict = {}
        # for pix in T_anomaly_vals_dict:
        #     vals = T_anomaly_vals_dict[pix]
        #     vals_mean = np.nanmean(vals)
        #     spatial_dict[pix] = vals_mean
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # plt.hist(T_anomaly_vals,bins=100)
        # plt.show()
        df['T_anomaly_mean'] = T_anomaly_vals
        df['NDVI_anomaly_mean'] = NDVI_anomaly_vals
        df = df[df['T_anomaly_mean']<=2]
        df = df[df['T_anomaly_mean']>=-2]
        bin_range = np.linspace(0,1,41)
        T_quantile_bins =[]
        AI_bins = np.linspace(0,2.5,26)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))
            # AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(len(T_quantile_bins))
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(len(T_quantile_bins))
        # plt.plot(T_quantile_bins)
        # plt.show()
        # print(len(T_quantile_bins))
        # print(len(AI_quantile_bins))
        # exit()
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(T_quantile_bins)
        # exi

        df_group_AI, _ = T.df_bin(df,'aridity_index',AI_bins)

        matrix = []
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i,'T_anomaly_mean',T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['NDVI_anomaly_mean'].tolist()
                rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                plt.scatter(bin_range[flag],y_label,c=rt_mean,vmin=-.8,vmax=.8,cmap='RdBu_r',marker='s')
                # print(flag,rt_mean)
                flag += 1
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_quantile')
        plt.colorbar()
        # plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        # plt.yticks(range(len(y_label_list))[::-1],y_label_list)
        # plt.tight_layout()
        # plt.show()
        outf = join(outdir,'matrix.pdf')
        plt.savefig(outf)
        plt.close()

        pass

    def normal_condition_AI_Tanomaly(self):
        outdir = join(self.this_class_png,'normal_condition_AI_Tanomaly')
        T.mkdir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        exit()
        df = df[df['aridity_index']<=3]
        # df = df.dropna(subset=['optimal_temp'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        NDVI_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            # NDVI_anomaly = row['NDVI_progress'].tolist()
            NDVI_anomaly = row['VPD-anomaly']
            # T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            # optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                continue
            NDVI_anomaly_vals.append(NDVI_anomaly)

            T_anomaly_reshape = np.reshape(T_anomaly, (-1, 6))
            drought_year_T_anomaly = T_anomaly_reshape[1]
            T_anomaly_mean = np.nanmean(drought_year_T_anomaly)
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_vals.append(T_anomaly_mean)
            # pix = row['pix']
            # if not pix in T_anomaly_vals_dict:
            #     T_anomaly_vals_dict[pix] = []
            # T_anomaly_vals_dict[pix].append(T_anomaly_mean)

            AI_vals.append(AI_val)
        # spatial_dict = {}
        # for pix in T_anomaly_vals_dict:
        #     vals = T_anomaly_vals_dict[pix]
        #     vals_mean = np.nanmean(vals)
        #     spatial_dict[pix] = vals_mean
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # plt.hist(T_anomaly_vals,bins=100)
        # plt.show()
        df['T_anomaly_mean'] = T_anomaly_vals
        df['NDVI_anomaly_mean'] = NDVI_anomaly_vals
        df = df[df['T_anomaly_mean']<=2]
        df = df[df['T_anomaly_mean']>=-2]
        bin_range = np.linspace(0,1,41)
        T_quantile_bins =[]
        AI_bins = np.linspace(0,2.5,26)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))
            # AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(len(T_quantile_bins))
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(len(T_quantile_bins))
        # plt.plot(T_quantile_bins)
        # plt.show()
        # print(len(T_quantile_bins))
        # print(len(AI_quantile_bins))
        # exit()
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(T_quantile_bins)
        # exi

        df_group_AI, _ = T.df_bin(df,'aridity_index',AI_bins)

        matrix = []
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i,'T_anomaly_mean',T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['NDVI_anomaly_mean'].tolist()
                rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                plt.scatter(bin_range[flag],y_label,c=rt_mean,vmin=-.8,vmax=.8,cmap='RdBu_r',marker='s')
                # print(flag,rt_mean)
                flag += 1
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_quantile')
        plt.colorbar()
        # plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        # plt.yticks(range(len(y_label_list))[::-1],y_label_list)
        # plt.tight_layout()
        # plt.show()
        outf = join(outdir,'matrix.pdf')
        plt.savefig(outf)
        plt.close()

        pass

    def sm_AI_Tanomaly(self):
        outdir = join(self.this_class_png,'sm_AI_Tanomaly')
        T.mkdir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        # exit()
        df = df[df['aridity_index']<=3]
        df = df.dropna(subset=['optimal_temp'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        NDVI_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            # NDVI_anomaly = row['NDVI_progress'].tolist()
            NDVI_anomaly = row['GLEAM-SMRoot-origin']
            # T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            # optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                continue
            NDVI_anomaly_vals.append(NDVI_anomaly)

            T_anomaly_reshape = np.reshape(T_anomaly, (-1, 6))
            drought_year_T_anomaly = T_anomaly_reshape[1]
            T_anomaly_mean = np.nanmean(drought_year_T_anomaly)
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_vals.append(T_anomaly_mean)
            # pix = row['pix']
            # if not pix in T_anomaly_vals_dict:
            #     T_anomaly_vals_dict[pix] = []
            # T_anomaly_vals_dict[pix].append(T_anomaly_mean)

            AI_vals.append(AI_val)
        # spatial_dict = {}
        # for pix in T_anomaly_vals_dict:
        #     vals = T_anomaly_vals_dict[pix]
        #     vals_mean = np.nanmean(vals)
        #     spatial_dict[pix] = vals_mean
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # plt.hist(T_anomaly_vals,bins=100)
        # plt.show()
        df['T_anomaly_mean'] = T_anomaly_vals
        df['NDVI_anomaly_mean'] = NDVI_anomaly_vals
        df = df[df['T_anomaly_mean']<=2]
        df = df[df['T_anomaly_mean']>=-2]
        bin_range = np.linspace(0,1,41)
        T_quantile_bins =[]
        AI_bins = np.linspace(0,2.5,26)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))
            # AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(len(T_quantile_bins))
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(len(T_quantile_bins))
        # plt.plot(T_quantile_bins)
        # plt.show()
        # print(len(T_quantile_bins))
        # print(len(AI_quantile_bins))
        # exit()
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(T_quantile_bins)
        # exi

        df_group_AI, _ = T.df_bin(df,'aridity_index',AI_bins)

        matrix = []
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i,'T_anomaly_mean',T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['NDVI_anomaly_mean'].tolist()
                rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                plt.scatter(bin_range[flag],y_label,c=rt_mean,vmin=0,vmax=.4,cmap='RdBu',marker='s')
                # print(flag,rt_mean)
                flag += 1
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_quantile')
        plt.colorbar()
        # plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        # plt.yticks(range(len(y_label_list))[::-1],y_label_list)
        # plt.tight_layout()
        # plt.show()
        outf = join(outdir,'matrix.pdf')
        plt.savefig(outf)
        plt.close()

        pass

    def sm_anomaly_AI_Tanomaly(self):
        outdir = join(self.this_class_png,'sm_anomaly_AI_Tanomaly')
        T.mkdir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        df = df[df['aridity_index']<=3]
        # df = df.dropna(subset=['optimal_temp'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        NDVI_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            # NDVI_anomaly = row['NDVI_progress'].tolist()
            # NDVI_anomaly = row['GLEAM-SMRoot-anomaly_detrend_progress'].tolist()
            # T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            # optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                continue
            # NDVI_anomaly_reshape = np.reshape(NDVI_anomaly, (-1, 6))
            # drought_year_NDVI_anomaly = NDVI_anomaly_reshape[1]
            NDVI_anomaly_mean = row['GLEAM-SMRoot-anomaly_detrend']
            NDVI_anomaly_vals.append(NDVI_anomaly_mean)

            T_anomaly_reshape = np.reshape(T_anomaly, (-1, 6))
            drought_year_T_anomaly = T_anomaly_reshape[1]
            T_anomaly_mean = np.nanmean(drought_year_T_anomaly)
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_vals.append(T_anomaly_mean)
            # pix = row['pix']
            # if not pix in T_anomaly_vals_dict:
            #     T_anomaly_vals_dict[pix] = []
            # T_anomaly_vals_dict[pix].append(T_anomaly_mean)

            AI_vals.append(AI_val)
        # spatial_dict = {}
        # for pix in T_anomaly_vals_dict:
        #     vals = T_anomaly_vals_dict[pix]
        #     vals_mean = np.nanmean(vals)
        #     spatial_dict[pix] = vals_mean
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # plt.hist(T_anomaly_vals,bins=100)
        # plt.show()
        df['T_anomaly_mean'] = T_anomaly_vals
        df['NDVI_anomaly_mean'] = NDVI_anomaly_vals
        df = df[df['T_anomaly_mean']<=2]
        df = df[df['T_anomaly_mean']>=-2]
        bin_range = np.linspace(0,1,41)
        T_quantile_bins =[]
        AI_bins = np.linspace(0,2.5,26)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))
            # AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(len(T_quantile_bins))
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(len(T_quantile_bins))
        # plt.plot(T_quantile_bins)
        # plt.show()
        # print(len(T_quantile_bins))
        # print(len(AI_quantile_bins))
        # exit()
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(T_quantile_bins)
        # exi

        df_group_AI, _ = T.df_bin(df,'aridity_index',AI_bins)

        matrix = []
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i,'T_anomaly_mean',T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['NDVI_anomaly_mean'].tolist()
                rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                plt.scatter(bin_range[flag],y_label,c=rt_mean,vmin=-1,vmax=1,cmap='RdBu',marker='s')
                # print(flag,rt_mean)
                flag += 1
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_quantile')
        plt.colorbar()
        # plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        # plt.yticks(range(len(y_label_list))[::-1],y_label_list)
        # plt.tight_layout()
        # plt.show()
        outf = join(outdir,'matrix.pdf')
        plt.savefig(outf)
        plt.close()

        pass

    def rt_AI_Tanomaly_line_plot(self):
        outdir = join(self.this_class_png,'rt_AI_Tanomaly_line_plot')
        T.mkdir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        # df = df[df['aridity_index']<=3]
        df = df.dropna(subset=['optimal_temp'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        NDVI_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            NDVI_anomaly = row['NDVI_progress'].tolist()
            # T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            # optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                T_anomaly_vals.append(np.nan)
                NDVI_anomaly_vals.append(np.nan)
                continue
            NDVI_anomaly_reshape = np.reshape(NDVI_anomaly, (-1, 6))
            drought_year_NDVI_anomaly = NDVI_anomaly_reshape[1]
            NDVI_anomaly_mean = np.nanmean(drought_year_NDVI_anomaly)
            NDVI_anomaly_vals.append(NDVI_anomaly_mean)

            T_anomaly_reshape = np.reshape(T_anomaly, (-1, 6))
            drought_year_T_anomaly = T_anomaly_reshape[1]
            T_anomaly_mean = np.nanmean(drought_year_T_anomaly)
            T_anomaly_vals.append(T_anomaly_mean)

            AI_vals.append(AI_val)
        df['T_anomaly_mean'] = T_anomaly_vals
        df['NDVI_anomaly_mean'] = NDVI_anomaly_vals
        df = df[df['T_anomaly_mean']<=2]
        df = df[df['T_anomaly_mean']>=-2]
        df = df.dropna(subset=['T_anomaly_mean','NDVI_anomaly_mean'],how='any')
        T_anomaly_vals = df['T_anomaly_mean'].tolist()
        NDVI_anomaly_vals = df['NDVI_anomaly_mean'].tolist()
        T_bin_range = np.linspace(0,1,41)
        T_quantile_bins =[]
        # AI_bins = np.linspace(0,2.5,26)
        AI_bins = np.linspace(0.7,2.5,18)
        # AI_bins = np.linspace(0,0.7,8)
        for b_i in T_bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))
        matrix_dict, x_ticks_list, y_ticks_list = T.df_bin_2d(df,'NDVI_anomaly_mean','T_anomaly_mean','aridity_index',T_quantile_bins,AI_bins)

        plt.figure(figsize=(7, 3.5))
        matrix = T.plot_df_bin_2d_matrix(matrix_dict,-.3,.3,x_ticks_list,y_ticks_list,cmap='RdBu',is_only_return_matrix=True)
        plt.imshow(matrix,cmap='RdBu',vmin=-.3,vmax=.3)
        plt.figure()
        x_mean = np.nanmean(matrix,axis=0)
        plt.plot(x_mean,marker='s',c='k')

        plt.show()
        pass

class Normal_year_Matrix:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Normal_year_Matrix', result_root_this_script, mode=2)
        # self.dff = join(self.this_class_arr, 'Drought_timing.df')
        self.dff = join(self.this_class_arr, 'normal_dataframe.df')
        pass

    def run(self):
        # self.gen_normal_df()
        # self.add_NDVI()
        # self.add_temperature_anomaly()
        self.NDVI_AI_Tanomaly()
        pass

    def gen_normal_df(self):
        drought_dff = Matrix().dff
        drought_df = T.load_df(drought_dff)
        drought_df_pix_group = T.df_groupby(drought_df,'pix')
        normal_year_list = []
        pix_list = []
        AI_list = []

        for pix in tqdm(drought_df_pix_group):
            df_i = drought_df_pix_group[pix]
            # T.print_head_n(df_i)

            aridity_index = df_i['aridity_index'].tolist()[0]
            # print(aridity_index);exit()
            drought_year_list = T.get_df_unique_val_list(df_i,'drought_year')
            for year in global_year_range_list:
                if year not in drought_year_list:
                    normal_year_list.append(year)
                    pix_list.append(pix)
                    AI_list.append(aridity_index)
        df_normal = pd.DataFrame()
        df_normal['pix'] = pix_list
        df_normal['normal_year'] = normal_year_list
        df_normal['aridity_index'] = AI_list
        T.print_head_n(df_normal)
        T.save_df(df_normal,self.dff)
        T.df_to_excel(df_normal, self.dff)

    def add_NDVI(self):
        # df = Load_dataframe()
        df = T.load_df(self.dff)
        vals_spatial_dict, data_name, valid_range = Load_Data().NDVI_anomaly_detrend()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_year = row['normal_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals, dtype=float)
            vals[vals > valid_range[1]] = np.nan
            vals[vals < valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals, gs, method='array')
            vals_gs_dict = T.dict_zip(year_list, vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        # T.print_head_n(df)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
        return df

    def add_temperature_anomaly(self):
        # df = Load_dataframe()
        df = T.load_df(self.dff)
        vals_spatial_dict, data_name, valid_range = Load_Data().Temperature_anomaly()
        # print(data_name)
        # exit()
        year_list = global_year_range_list
        gs = global_gs
        vals_drought_year_mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_year = row['normal_year']
            # print(drought_year)
            # exit()
            vals = vals_spatial_dict[pix]
            vals = np.array(vals, dtype=float)
            vals[vals > valid_range[1]] = np.nan
            vals[vals < valid_range[0]] = np.nan
            # vals[vals<0] = np.nan
            vals_gs = T.monthly_vals_to_annual_val(vals, gs, method='array')
            vals_gs_dict = T.dict_zip(year_list, vals_gs)
            vals_drought_year = vals_gs_dict[drought_year]
            vals_drought_year_mean = np.nanmean(vals_drought_year)
            vals_drought_year_mean_list.append(vals_drought_year_mean)
        df[f'{data_name}'] = vals_drought_year_mean_list
        # T.print_head_n(df)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)
        return df

    def NDVI_AI_Tanomaly(self):
        outdir = join(self.this_class_png,'NDVI_AI_Tanomaly')
        T.mkdir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df, 10)
        # pause()
        df = df[df['aridity_index']<=3]
        df = df.dropna(subset=['NDVI-anomaly_detrend'])
        # T_anomaly = df['Temperature-anomaly_detrend_progress']
        T_anomaly_vals = []
        NDVI_anomaly_vals = []
        AI_vals = []
        T_anomaly_vals_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            # T_anomaly = row['Temperature-anomaly_detrend_progress'].tolist()
            # NDVI_anomaly = row['NDVI_progress'].tolist()
            # T_anomaly = row['Temperature-origin_progress'].tolist()
            AI_val = row['aridity_index']
            # optimal_temp = row['optimal_temp']
            # if np.isnan(optimal_temp):
            #     continue
            # print(optimal_temp)
            if np.isnan(AI_val):
                continue
            NDVI_anomaly_mean = row['NDVI-anomaly_detrend']
            # print(NDVI_anomaly_mean);exit()
            NDVI_anomaly_vals.append(NDVI_anomaly_mean)

            T_anomaly_mean = row['Temperature-anomaly']
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            # T_anomaly_mean = np.nanmax(drought_year_T_anomaly) - optimal_temp
            T_anomaly_vals.append(T_anomaly_mean)
            # pix = row['pix']
            if not pix in T_anomaly_vals_dict:
                T_anomaly_vals_dict[pix] = []
            T_anomaly_vals_dict[pix].append(T_anomaly_mean)

            AI_vals.append(AI_val)
        # spatial_dict = {}
        # for pix in T_anomaly_vals_dict:
        #     vals = T_anomaly_vals_dict[pix]
        #     vals_mean = np.nanmean(vals)
        #     spatial_dict[pix] = vals_mean
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # plt.hist(T_anomaly_vals,bins=100)
        # plt.show()
        df['T_anomaly_mean'] = T_anomaly_vals
        df['NDVI_anomaly_mean'] = NDVI_anomaly_vals
        df = df[df['T_anomaly_mean']<=2]
        df = df[df['T_anomaly_mean']>=-2]
        bin_range = np.linspace(0,1,41)
        T_quantile_bins =[]
        AI_bins = np.linspace(0,2.5,26)
        for b_i in bin_range:
            T_quantile_bins.append(np.quantile(T_anomaly_vals,b_i))
            # AI_quantile_bins.append(np.quantile(AI_vals,b_i))
        # print(len(T_quantile_bins))
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(len(T_quantile_bins))
        # plt.plot(T_quantile_bins)
        # plt.show()
        # print(len(T_quantile_bins))
        # print(len(AI_quantile_bins))
        # exit()
        # T_quantile_bins = T.drop_repeat_val_from_list(T_quantile_bins)
        # print(T_quantile_bins)
        # exit()

        df_group_AI, _ = T.df_bin(df,'aridity_index',AI_bins)

        matrix = []
        y_label_list = []
        plt.figure(figsize=(7, 3.5))
        for name_AI, df_group_AI_i in df_group_AI:
            matrix_i = []
            y_label = (name_AI[0].left + name_AI[0].right) / 2
            y_label = np.round(y_label, 2)
            y_label_list.append(y_label)
            x_label_list = []

            df_group_T, _ = T.df_bin(df_group_AI_i,'T_anomaly_mean',T_quantile_bins)
            # if len(df_group_T) != len(T_quantile_bins)-1:
            #     continue
            # print(len(T_quantile_bins))
            flag = 0

            for name_T, df_group_T_i in df_group_T:
                rt = df_group_T_i['NDVI_anomaly_mean'].tolist()
                rt_mean = np.nanmean(rt)
                matrix_i.append(rt_mean)
                # print(rt_mean)
                x_label = (name_T[0].left + name_T[0].right) / 2
                x_label = np.round(x_label, 2)
                x_label_list.append(x_label)
                plt.scatter(bin_range[flag],y_label,c=rt_mean,vmin=-.3,vmax=.3,cmap='RdBu',marker='s')
                # print(flag,rt_mean)
                flag += 1
        plt.ylabel('AI')
        plt.xlabel('T_anomaly_quantile')
        plt.colorbar()
        # plt.xticks(range(len(x_label_list)),x_label_list,rotation=90)
        # plt.yticks(range(len(y_label_list))[::-1],y_label_list)
        # plt.tight_layout()
        # plt.show()
        outf = join(outdir,'NDVI_AI_Tanomaly.pdf')
        plt.savefig(outf)
        plt.close()

        pass

def main():
    Matrix().run()
    # Normal_year_Matrix().run()

    pass
    main()



if __name__ == '__main__':
    main()