# coding=utf-8
import matplotlib.pyplot as plt

from meta_info import *

result_root_this_script = join(results_root, 'matrix')


class Matrix:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Matrix', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'Drought_timing.df')
        pass

    def run(self):
        # self.copy_df()
        # self.rt_AI_Topt()
        self.rt_AI_Tanomaly()
        pass
    def copy_df(self):
        import statistic
        if isfile(self.dff):
            print('already exists: ', self.dff)
            print('press enter to overwrite')
            pause()
            pause()
            pause()
        dff = statistic.Drought_timing().dff
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
                plt.scatter(bin_range[flag],y_label,c=rt_mean,vmin=-.3,vmax=.3,cmap='RdBu',marker='s')
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

def main():
    Matrix().run()

    pass



if __name__ == '__main__':
    main()