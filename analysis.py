# coding=utf-8
import matplotlib.pyplot as plt

from meta_info import *
result_root_this_script = join(results_root, 'analysis')
import xymap


class Water_energy_limited_area:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir('Water_energy_limited_area',
                                                                                       result_root_this_script, mode=2)
        pass

    def run(self):
        # self.kendall_corr_all_vars()
        self.Ecosystem_Limited_Index_p()
        # self.Ecosystem_Limited_Index()
        pass

    def kendall_corr_all_vars(self):
        outdir = join(self.this_class_arr, 'kendall_corr')
        var_1 = 'GLEAM-ET'
        var_21 = 'ERA-SM'
        var_22 = 'CCI-SM'
        var_3 = 'Temperature'
        var_31 = 'Radiation'
        # self.kendall_corr(var_1,var_21,outdir)
        # self.kendall_corr(var_1,var_22,outdir)
        # self.kendall_corr(var_1,var_3,outdir)
        self.kendall_corr(var_1, var_31, outdir)

    def kendall_corr(self, var_1, var_2, outdir):
        T.mk_dir(outdir)
        outf = join(outdir, f'{var_1}_{var_2}.df')
        spatial_dict_1 = Meta_information().load_data(var_1)
        spatial_dict_2 = Meta_information().load_data(var_2)
        spatial_dict_corr = {}
        spatial_dict_corr_p = {}
        for pix in tqdm(spatial_dict_1):
            if not pix in spatial_dict_2:
                continue
            val1 = spatial_dict_1[pix]
            val2 = spatial_dict_2[pix]
            r, p = T.nan_correlation(val1, val2, method='kendall')
            spatial_dict_corr[pix] = r
            spatial_dict_corr_p[pix] = p

        spatial_dict_all = {
            f'{var_1}_{var_2}_r': spatial_dict_corr,
            f'{var_1}_{var_2}_p': spatial_dict_corr_p,
        }
        df = T.spatial_dics_to_df(spatial_dict_all)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def Ecosystem_Limited_Index(self):
        var_1 = 'GLEAM-ET'
        var_21 = 'ERA-SM'
        var_22 = 'CCI-SM'
        # var_3 = 'Temperature'
        var_31 = 'Radiation'
        fdir = join(self.this_class_arr, 'kendall_corr')
        outdir = join(self.this_class_tif, 'ELI')
        T.mk_dir(outdir)
        # outf = join(outdir, f'{var_1}_{var_21}_{var_3}.tif')
        outf = join(outdir, f'{var_1}_{var_21}_{var_31}.tif')
        ELI_equation = 'ELI = corr(ET,SM) - corr(ET,T)'
        dff1 = join(fdir, f'{var_1}_{var_21}.df')
        # dff2 = join(fdir,f'{var_1}_{var_3}.df')
        dff2 = join(fdir, f'{var_1}_{var_31}.df')
        df1 = T.load_df(dff1)
        df2 = T.load_df(dff2)
        spatial_dict1 = T.df_to_spatial_dic(df1, f'{var_1}_{var_21}_r')
        # spatial_dict2 = T.df_to_spatial_dic(df2,f'{var_1}_{var_3}_r')
        spatial_dict2 = T.df_to_spatial_dic(df2, f'{var_1}_{var_31}_r')

        ELI_spatial_dict = {}
        for pix in tqdm(spatial_dict1):
            if not pix in spatial_dict2:
                continue
            val1 = spatial_dict1[pix]
            val2 = spatial_dict2[pix]
            ELI = val1 - val2
            ELI_spatial_dict[pix] = ELI

        DIC_and_TIF().pix_dic_to_tif(ELI_spatial_dict, outf)

    def Ecosystem_Limited_Index_p(self):
        var_1 = 'GLEAM-ET'
        var_21 = 'ERA-SM'
        var_22 = 'CCI-SM'
        var_3 = 'Temperature'
        # var_31 = 'Radiation'
        fdir = join(self.this_class_arr, 'kendall_corr')
        outdir = join(self.this_class_tif, 'ELI')
        T.mk_dir(outdir)
        # outf = join(outdir, f'{var_1}_{var_21}_{var_31}_p.tif')
        outf = join(outdir, f'{var_1}_{var_21}_{var_3}_p.tif')
        ELI_equation = 'ELI = corr(ET,SM) - corr(ET,T)'
        dff1 = join(fdir, f'{var_1}_{var_21}.df')
        dff2 = join(fdir, f'{var_1}_{var_3}.df')
        # dff2 = join(fdir,f'{var_1}_{var_31}.df')
        df1 = T.load_df(dff1)
        df2 = T.load_df(dff2)
        spatial_dict1 = T.df_to_spatial_dic(df1, f'{var_1}_{var_21}_p')
        spatial_dict2 = T.df_to_spatial_dic(df2, f'{var_1}_{var_3}_p')
        # spatial_dict2 = T.df_to_spatial_dic(df2,f'{var_1}_{var_31}_p')

        ELI_spatial_dict = {}
        for pix in tqdm(spatial_dict1):
            if not pix in spatial_dict2:
                continue
            val1 = spatial_dict1[pix]
            val2 = spatial_dict2[pix]
            if val1 > 0.05 or val2 > 0.05:
                ELI = 1
            else:
                ELI = 0
            ELI_spatial_dict[pix] = ELI

        DIC_and_TIF().pix_dic_to_tif(ELI_spatial_dict, outf)


class Water_energy_limited_area_daily:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
            'Water_energy_limited_area_daily',
            result_root_this_script, mode=2)
        pass

    def run(self):
        # self.ELI()
        # self.ELI_perpix()
        self.anomaly()

        pass

    def load_data(self, year, mon):
        year = int(year)
        year = str(year)
        mon = int(mon)
        mon = f'{mon:02d}'

        ET_path = join(data_root, 'GLEAM_daily', 'perpix', year, 'Et', mon)
        SMsurf_path = join(data_root, 'GLEAM_daily', 'perpix', year, 'SMsurf', mon)
        SMroot_path = join(data_root, 'GLEAM_daily', 'perpix', year, 'SMroot', mon)
        T_path = join(data_root, 'ERA_daily_Tair', 'perpix', year, mon)

        ET_dict = T.load_npy_dir(ET_path)
        SMsurf_dict = T.load_npy_dir(SMsurf_path)
        SMroot_dict = T.load_npy_dir(SMroot_path)
        T_dict = T.load_npy_dir(T_path)
        return ET_dict, SMsurf_dict, SMroot_dict, T_dict

    def ELI(self):
        outdir = join(self.this_class_tif, 'ELI')
        T.mk_dir(outdir)
        year_list = list(range(global_start_year, global_end_year + 1))
        mon_list = list(range(1, 13))
        for year in tqdm(year_list):
            for mon in mon_list:
                outf = join(outdir, f'{year}{mon:02d}.tif')
                ET_dict, SMsurf_dict, SMroot_dict, T_dict = self.load_data(year, mon)
                spatial_dict = {}
                for pix in ET_dict:
                    r, c = pix
                    if r > 180:
                        continue
                    if not pix in SMsurf_dict or not pix in SMroot_dict or not pix in T_dict:
                        continue
                    ET = ET_dict[pix]
                    SMsurf = SMsurf_dict[pix]
                    # SMroot = SMroot_dict[pix]
                    Temp = T_dict[pix]
                    if T.is_all_nan(ET):
                        continue
                    ET_sm_corr, _ = T.nan_correlation(ET, SMsurf)
                    ET_Temp_corr, _ = T.nan_correlation(ET, Temp)
                    ELI = ET_sm_corr - ET_Temp_corr
                    spatial_dict[pix] = ELI
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                DIC_and_TIF().arr_to_tif(arr, outf)

    def ELI_perpix(self):
        fdir = join(self.this_class_tif, 'ELI')
        outdir = join(self.this_class_arr, 'ELI', 'perpix', '1982-2015')
        T.mk_dir(outdir, force=1)
        Pre_Process().data_transform(fdir, outdir)

        pass

    def anomaly(self):
        fdir = join(self.this_class_arr, 'ELI', 'perpix', '1982-2015')
        outdir = join(self.this_class_arr, 'ELI', 'anomaly', '1982-2015')
        T.mk_dir(outdir, force=1)
        Pre_Process().cal_anomaly(fdir, outdir)

        pass


class Pick_Drought_Events:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_Drought_Events', result_root_this_script, mode=2)
        self.threshold = -2

    def run(self):
        # self.pick_normal_drought_events()
        # self.pick_normal_hot_events()
        # self.pick_single_events(year_range_str)
        # self.check_drought_events()
        # self.drought_timing()
        self.gen_dataframe()
        pass

    def pick_normal_hot_events(self):
        outdir = join(self.this_class_arr, 'normal_hot_events')
        T.mk_dir(outdir)
        threshold_quantile = 90
        # gs_dict = Growing_season().longterm_growing_season()
        t_anomaly_dic,_ = Load_Data().Temperature_origin_detrend()
        drought_events_dir = join(self.this_class_arr, 'picked_events')
        for f in T.listdir(drought_events_dir):
            scale = f.split('.')[0]
            fpath = join(drought_events_dir, f)
            drought_events_dict = T.load_npy(fpath)
            hot_dic = {}
            normal_dic = {}
            for pix in tqdm(drought_events_dict,desc=f'{scale}'):
                spi_drought_year = drought_events_dict[pix]
                temp_anomaly = t_anomaly_dic[pix]
                # if not pix in global_gs_dict:
                #     continue
                # gs_mon = global_gs_dict[pix]
                # gs_mon = list(gs_mon)
                gs_mon = global_gs
                T_annual_val = T.monthly_vals_to_annual_val(temp_anomaly, gs_mon, method='mean')
                T_quantile = np.percentile(T_annual_val, threshold_quantile)
                hot_index_True_False = T_annual_val > T_quantile
                hot_years = []
                for i, val in enumerate(hot_index_True_False):
                    if val == True:
                        hot_years.append(i + global_start_year)
                hot_years = set(hot_years)
                # print(hot_years)
                # exit()
                hot_drought_year = []
                spi_drought_year_spare = []
                for dr in spi_drought_year:
                    if dr in hot_years:
                        hot_drought_year.append(dr)
                    else:
                        spi_drought_year_spare.append(dr)
                hot_dic[pix] = hot_drought_year
                normal_dic[pix] = spi_drought_year_spare
            hot_outf = join(outdir, f'hot-drought_{scale}.npy')
            normal_outf = join(outdir, f'normal-drought_{scale}.npy')
            T.save_npy(hot_dic, hot_outf)
            T.save_npy(normal_dic, normal_outf)

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        T.mk_dir(outdir)
        threshold = self.threshold
        SPI_dict_all,_ = Load_Data().SPI()

        for scale in global_selected_spi_list:
            SPI_dict = SPI_dict_all[scale]
            events_dic = {}
            params_list = []
            for pix in tqdm(SPI_dict,desc=f'{scale}'):
                vals = SPI_dict[pix]
                vals = np.array(vals)
                params = (vals, threshold)
                params_list.append(params)
                events_list = self.kernel_find_drought_period(params)
                if len(events_list) == 0:
                    continue
                drought_year_list = []
                for drought_range in events_list:
                    min_index = T.pick_min_indx_from_1darray(vals, drought_range)
                    drought_year = min_index // 12 + global_start_year
                    drought_year_list.append(drought_year)
                drought_year_list = np.array(drought_year_list)
                events_dic[pix] = drought_year_list
            outf = join(outdir, f'{scale}')
            T.save_npy(events_dic, outf)

    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        threshold = params[1]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:  # SPEI
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 0:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []

        flag = 0
        events_list = []
        # 不取两个端点
        for i in events:
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                new_i.append(jj)
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()
            # print(vals)

            # if 0 in new_i:
            # SPEI
            min_val = min(vals)
            if min_val < -99999:
                continue

            events_list.append(new_i)
        return events_list

    def __get_drought_events(self):
        outdir = join(self.this_class_arr, 'drought_events_df')
        T.mk_dir(outdir)
        outf = join(outdir, 'drought_events.df')
        if isfile(outf):
            df = T.load_df(outf)
            return df
        drought_events_dir = join(self.this_class_arr, 'normal_hot_events')
        spatial_dict_all = {}
        for f in T.listdir(drought_events_dir):
            fpath = join(drought_events_dir, f)
            var_i = f.split('.')[0]
            spatial_dict = T.load_npy(fpath)
            spatial_dict_all[var_i] = spatial_dict
        df = T.spatial_dics_to_df(spatial_dict_all)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        return df
        pass

    def check_drought_events(self):
        drought_events_df = self.__get_drought_events()
        T.print_head_n(drought_events_df, 10)
        pass

    def pick_single_events(self, year_range_str):
        outdir = join(self.this_class_arr, 'pick_single_events/{}'.format(year_range_str))
        T.mk_dir(outdir, force=True)
        outf = join(outdir, 'single_events.df')
        hot_drought_f = join(self.this_class_arr, f'picked_events/spi_and_hot_12/{year_range_str}/hot_drought_12.npy')
        spi_drought_f = join(self.this_class_arr, f'picked_events/spi_and_hot_12/{year_range_str}/spi_drought_12.npy')
        hot_drought_dic = T.load_npy(hot_drought_f)
        spi_drought_dic = T.load_npy(spi_drought_f)
        pix_list = DIC_and_TIF().void_spatial_dic()
        spatial_dic_dic = {}
        for pix in pix_list:
            spatial_dic_dic[pix] = {}
        for pix in pix_list:
            if not pix in hot_drought_dic:
                continue
            spatial_dic_dic[pix]['hot_drought'] = hot_drought_dic[pix]

        for pix in pix_list:
            if not pix in spi_drought_dic:
                continue
            spatial_dic_dic[pix]['dry_drought'] = spi_drought_dic[pix]
        single_events_spatial_dic = {}
        for pix in tqdm(spatial_dic_dic):
            dic = spatial_dic_dic[pix]
            if len(dic) == 0:
                continue
            drought_years_list = []
            for dtype in dic:
                drought_years = dic[dtype]
                for year in drought_years:
                    drought_years_list.append(year)
            # print(dic)
            drought_years_list = T.drop_repeat_val_from_list(drought_years_list)
            drought_years_list.sort()
            # print('drought_years_list',drought_years_list)
            single_events_list = self.__pick_single_events(drought_years_list)
            # print('single_events_list',single_events_list)
            single_events_dic = {}
            for dtype in dic:
                drought_years = dic[dtype]
                single_event = []
                for year in single_events_list:
                    if year in drought_years:
                        single_event.append(year)
                single_event = np.array(single_event, dtype=int)
                if len(single_event) == 0:
                    single_events_dic[dtype] = np.nan
                else:
                    single_events_dic[dtype] = single_event
            single_events_spatial_dic[pix] = single_events_dic
        df = T.dic_to_df(single_events_spatial_dic, 'pix')
        # self.shasum_variable(df)
        # exit()

        col_list = df.columns.to_list()
        col_list.remove('pix')
        df = df.dropna(how='all', subset=col_list)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def shasum_variable(self, variable):
        readable_hash = hashlib.sha256(str(variable).encode('ascii')).hexdigest()
        print(readable_hash)
        return readable_hash

    def __pick_single_events(self, drought_year_list):
        n = 4
        single_events_list = []
        for i in range(len(drought_year_list)):
            year = drought_year_list[i]
            if i - 1 < 0:  # first drought event
                if len(drought_year_list) == 1:
                    single_events_list.append(year)
                    break
                if year + n <= drought_year_list[i + 1]:
                    single_events_list.append(year)
                continue
            if i + 1 >= len(drought_year_list):  # the last drought event
                if drought_year_list[i] - drought_year_list[i - 1] >= n:
                    single_events_list.append(drought_year_list[i])
                break
            if drought_year_list[i] - drought_year_list[i - 1] >= n and drought_year_list[i] + n <= drought_year_list[
                i + 1]:  # middle drought events
                single_events_list.append(drought_year_list[i])
        return single_events_list

    def add_drought_season(self,df):
        season_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            mon = row['drought_mon']
            season = global_season_mon_dict[mon]
            season_list.append(season)
        df['drought_season'] = season_list
        return df


    def drought_timing(self):
        outdir = join(self.this_class_arr, 'drought_timing')
        T.mk_dir(outdir)
        threshold = self.threshold

        SPI_dict_all,_ = Load_Data().SPI()
        for scale in global_selected_spi_list:
            SPI_dict = SPI_dict_all[scale]
            events_dic = {}
            events_mon_dic = {}
            params_list = []
            for pix in tqdm(SPI_dict):
                vals = SPI_dict[pix]
                vals = np.array(vals)
                params = (vals, threshold)
                params_list.append(params)
                events_list = self.kernel_find_drought_period(params)
                if len(events_list) == 0:
                    continue
                drought_year_list = []
                drought_month_list = []
                for drought_range in events_list:
                    min_index = T.pick_min_indx_from_1darray(vals, drought_range)
                    drought_year = min_index // 12 + global_start_year
                    drought_month = min_index % 12 + 1
                    drought_year_list.append(drought_year)
                    drought_month_list.append(drought_month)
                drought_year_list = np.array(drought_year_list)
                drought_month_list = np.array(drought_month_list)
                events_dic[pix] = drought_year_list
                events_mon_dic[pix] = drought_month_list
            outf_year = join(outdir, f'{scale}_drought_year.npy')
            outf_mon = join(outdir, f'{scale}_drought_mon.npy')
            T.save_npy(events_dic, outf_year)
            T.save_npy(events_mon_dic, outf_mon)
            pass


    def gen_dataframe(self):
        outdir = join(self.this_class_arr,'drought_dataframe')
        T.mk_dir(outdir)
        drought_events_dir = join(self.this_class_arr, 'normal_hot_events')
        drought_timing_dir = join(self.this_class_arr,'drought_timing')
        # drought_year_f = join(drought_timing_dir,'drought_year.npy')
        # drought_mon_f = join(drought_timing_dir,'drought_mon.npy')


        pix_list = []
        drought_year_list = []
        drought_type_list = []
        drought_scale_list = []
        drought_year_dict_all = {}
        drought_mon_dict_all = {}
        for scale in global_selected_spi_list:
            drought_year_f = join(drought_timing_dir,f'{scale}_drought_year.npy')
            drought_mon_f = join(drought_timing_dir,f'{scale}_drought_mon.npy')
            drought_year_dict = T.load_npy(drought_year_f)
            drought_mon_dict = T.load_npy(drought_mon_f)
            drought_year_dict_all[scale] = drought_year_dict
            drought_mon_dict_all[scale] = drought_mon_dict

        for f in tqdm(T.listdir(drought_events_dir)):
            fpath = join(drought_events_dir, f)
            var_i = f.split('.')[0]
            drought_type = var_i.split('_')[0]
            scale = var_i.split('_')[1]
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                events = spatial_dict[pix]
                for e in events:
                    pix_list.append(pix)
                    drought_year_list.append(e)
                    drought_type_list.append(drought_type)
                    drought_scale_list.append(scale)
        # exit()
        df = pd.DataFrame()
        df['pix'] = pix_list
        df['drought_year'] = drought_year_list
        df['drought_type'] = drought_type_list
        df['drought_scale'] = drought_scale_list
        # T.print_head_n(df)
        # exit()
        # add drought timing
        # drought_timing_year_list = []
        drought_mon_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year = row['drought_year']
            scale = row['drought_scale']
            drought_year = drought_year_dict_all[scale][pix]
            drought_year = list(drought_year)
            drought_mon = drought_mon_dict_all[scale][pix]
            drought_mon = list(drought_mon)
            drought_year_index = drought_year.index(year)
            drought_mon_i = drought_mon[drought_year_index]
            drought_mon_list.append(drought_mon_i)

        df['drought_mon'] = drought_mon_list
        df = self.add_drought_season(df)
        df = df.sort_values(by=['pix','drought_type','drought_year'])
        # re index dataframe
        df = df.reset_index(drop=True)

        outf = join(outdir,'drought_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)


class Phenology:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Phenology', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.hants_interpolation()
        # self.pick_phenology()
        # self.phenology_df()
        self.check_phenology()
        pass

    def hants_interpolation(self):
        NDVI_bi_weekly_tif_dir = join(data_root, 'NDVI4g/bi_weekly_05')
        fdir = join(data_root,'NDVI4g/per_pix_biweekly')
        outdir = join(self.this_class_arr,'hants_interpolation')
        T.mk_dir(outdir)

        dates_list = []
        for f in T.listdir(NDVI_bi_weekly_tif_dir):
            date_str = f.split('.')[0]
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:])
            date_obj = datetime.datetime(year,month,day)
            dates_list.append(date_obj)
        params_list = []
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            params_list.append([fpath,outpath,dates_list])
            self.kernel_hants_interpolation([fpath,outpath,dates_list])
        # MULTIPROCESS(self.kernel_hants_interpolation, params_list).run(process=2)

        pass

    def kernel_hants_interpolation(self,params):
        fpath,outpath,dates_list = params
        spatial_dict = T.load_npy(fpath)
        hants_spatial_dict = {}
        for pix in tqdm(spatial_dict,desc=fpath):
            r, c = pix
            if r > 120:
                continue
            vals = spatial_dict[pix]
            vals = np.array(vals, dtype=float)
            vals[vals > 10000] = np.nan
            vals[vals < 0] = np.nan
            if T.is_all_nan(vals):
                continue
            try:
                results_dict = HANTS().hants_interpolate(vals, dates_list, (0, 10000), nan_value=0)
                hants_spatial_dict[pix] = results_dict
            except:
                continue
        T.save_npy(hants_spatial_dict, outpath)

    def pick_phenology(self):
        hants_fdir = join(self.this_class_arr,'hants_interpolation')
        outdir = join(self.this_class_arr,'phenology')
        T.mk_dir(outdir)
        year_list = global_year_range_list
        df_dict = {}
        for f in T.listdir(hants_fdir):
            fpath = join(hants_fdir,f)
            hants_spatial_dict = T.load_npy(fpath)
            if len(hants_spatial_dict) == 0:
                continue
            for pix in tqdm(hants_spatial_dict,desc=fpath):
                annual_vals_dict = hants_spatial_dict[pix]
                for year in annual_vals_dict:
                    vals = annual_vals_dict[year]
                    if T.is_all_nan(vals):
                        continue
                    try:
                        phenology_info_dict = self.pick_early_peak_late_dormant_period(vals)
                    except:
                        continue
                    if not pix in df_dict:
                        df_dict[pix] = {}
                    df_dict[pix][year] = phenology_info_dict
        outf = join(outdir,'phenology.npy')
        T.save_npy(df_dict,outf)


    def phenology_df(self):
        fpath = join(self.this_class_arr,'phenology/phenology.npy')
        outdir = join(self.this_class_arr,'phenology_df')
        outf = join(outdir,'phenology_df.df')
        phenology_dict = T.load_npy(fpath)
        col_list = []
        year_list = []
        for pix in phenology_dict:
            phenology_dict_i = phenology_dict[pix]
            year_list = sorted(phenology_dict_i.keys())
            col_list = list(phenology_dict_i[year_list[0]].keys())
            break
        spatial_dict = {}
        for pix in tqdm(phenology_dict):
            phenology_dict_i = phenology_dict[pix]
            col_dict = {}
            for col in col_list:
                year_dict = {}
                for year in year_list:
                    if not year in phenology_dict_i:
                        continue
                    year_dict[year] = phenology_dict_i[year][col]
                col_dict[col] = year_dict
            spatial_dict[pix] = col_dict
        df = T.dic_to_df(spatial_dict,'pix')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        T.open_path_and_file(outdir)

    def pick_early_peak_late_dormant_period(self,NDVI_daily,threshold=0.3):
        '''
        :param NDVI_daily: 365-day NDVI time series
        :param threshold: SOS and EOS threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: details of phenology
        '''
        peak = np.argmax(NDVI_daily)
        if peak == 0 or peak == (len(NDVI_daily)-1):
            raise
        try:
            early_start = self.__search_SOS(NDVI_daily, peak, threshold)
            late_end = self.__search_EOS(NDVI_daily, peak, threshold)
        except:
            early_start = np.nan
            late_end = np.nan
        # method 1
        # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak) # unstable
        # method 2
        early_end, late_start = self.__median_early_late(NDVI_daily,early_start,late_end,peak) # choose the median value before and after the peak

        early_length = early_end - early_start
        mid_length = late_start - early_end
        late_length = late_end - late_start
        dormant_length = 365 - (late_end - early_start)

        result = {
            'early_length':early_length,
            'mid_length':mid_length,
            'late_length':late_length,
            'dormant_length':dormant_length,
            'early_start':early_start,
            'early_start_mon':self.__doy_to_month(early_start),
            'early_end':early_end,
            'early_end_mon':self.__doy_to_month(early_end),
            'peak':peak,
            'peak_mon':self.__doy_to_month(peak),
            'late_start':late_start,
            'late_start_mon':self.__doy_to_month(late_start),
            'late_end':late_end,
            'late_end_mon':self.__doy_to_month(late_end),
            # 'growing_season':list(range(early_start,late_end)),
            # 'growing_season_mon':[self.__doy_to_month(i) for i in range(early_start,late_end)],
            # 'dormant_season':[i for i in range(0,early_start)]+[i for i in range(late_end,365)],
            # 'dormant_season_mon':[self.__doy_to_month(i) for i in range(0,early_start)]+[self.__doy_to_month(i) for i in range(late_end,365)],
        }
        return result

    def check_phenology(self):
        fpath = join(self.this_class_arr,'phenology_df/phenology_df.df')
        df = T.load_df(fpath)
        cols = list(df.columns)
        print(cols)
        spatial_dict = {}
        for i,row in df.iterrows():
            pix = row['pix']
            early_start = row['early_start']
            early_start_dict = dict(early_start)
            vals_list = []
            for year in early_start_dict:
                val = early_start_dict[year]
                vals_list.append(val)
            mean = np.nanmean(vals_list)
            # print(early_start_dict)
            # exit()
            spatial_dict[pix] = mean
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()

    def __doy_to_month(self,doy):
        '''
        :param doy: day of year
        :return: month
        '''
        base = datetime.datetime(2000,1,1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day
        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

    def __interp(self, vals):
        '''
        :param vals: bi-weekly NDVI values
        :return: 365-day NDVI time series with linear interpolation
        '''
        inx = list(range(len(vals)))
        iny = vals
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)
        return x_new, y_new

    def __search_SOS(self, vals, maxind, threshold_i):
        '''
        :param vals: 365-day NDVI time series
        :param maxind: the index of the peak value
        :param threshold_i: threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: the index of the Start of Season (SOS)
        '''
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        if left_min < 2000: # for NDVI, 2000 is equivalent to 0.2
            left_min = 2000
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_EOS(self, vals, maxind, threshold_i):
        '''
        :param vals: 365-day NDVI time series
        :param maxind: the index of the peak value
        :param threshold_i: threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: the index of the End of Season (EOS)
        '''
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        if right_min < 2000: # for NDVI, 2000 is equivalent to 0.2
            right_min = 2000
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold: # stop search when the value is lower than threshold
                break
        return ind

    def __slope_early_late(self,vals,sos,eos,peak):
        slope_left = []
        for i in range(sos,peak):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_left.append(slope_i)

        slope_right = []
        for i in range(peak,eos):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_right.append(slope_i)

        max_ind = np.argmax(slope_left) + sos
        min_ind = np.argmin(slope_right) + peak

        return max_ind, min_ind

    def __median_early_late(self,vals,sos,eos,peak):
        '''
        :param vals: 365-day NDVI time series
        :param sos: the index of the Start of Season (SOS)
        :param eos: the index of the End of Season (EOS)
        :param peak: the index of the peak index
        :return: the index of the early end and late start
        '''
        median_left = int((peak-sos)/2.)
        median_right = int((eos - peak)/2.)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind


class Long_term_correlation:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Long_term_correlation', result_root_this_script, mode=2)
        pass


    def run(self):
        # ---------------------------------------
        # A_data_obj = Load_Data().SPI
        # A_data_obj = Load_Data().Temperature_anomaly_detrend
        # A_data_obj = Load_Data().Precipitation_anomaly_detrend
        # A_data_obj = Load_Data().FAPAR_anomaly_detrend
        # A_data_obj = Load_Data().Srad_anomaly_detrend
        # # --------------
        # B_data_obj = Load_Data().NDVI_anomaly_detrend
        # # ---------------------------------------
        # self.A_vs_B_seasonal_correlation(A_data_obj,B_data_obj)

        # pheno_str = 'early_start'
        # pheno_str = 'late_end'
        # self.phenology_correlation(pheno_str,A_data_obj)

        # # ---------------------------------------
        # self.correlation_tif()
        self.plot_correlation_tif()
        # self.plot_correlation_bar()
        # self.plot_correlation_bar_all_region()
        # self.correlation_statistic()




    def A_vs_B_seasonal_correlation(self,A_data_obj,B_data_obj):
        pix_list = DIC_and_TIF().void_spatial_dic()
        A_dict, A_var_name = A_data_obj()
        B_dict, B_var_name = B_data_obj()
        outdir = join(self.this_class_arr,'seasonal_correlation',f'{A_var_name}_vs_{B_var_name}')
        T.mk_dir(outdir,force=True)

        drought_season_dict = global_drought_season_dict
        for season in drought_season_dict:
            corr_dict = {}
            for pix in tqdm(pix_list,desc=season):
                r_,c_ = pix
                if r_ > 120:
                    continue
                if not pix in A_dict:
                    continue
                if not pix in B_dict:
                    continue
                vals_A = A_dict[pix]
                vals_B = B_dict[pix]
                gs = drought_season_dict[season]
                vals_A_annual = T.monthly_vals_to_annual_val(vals_A,gs)
                vals_B_annual = T.monthly_vals_to_annual_val(vals_B,gs)
                try:
                    a,b,r,p = T.nan_line_fit(vals_A_annual,vals_B_annual)
                except:
                    a,b,r,p = np.nan,np.nan,np.nan,np.nan
                corr_dict[pix] = {
                    'a':a,
                    'b':b,
                    'r':r,
                    'p':p,
                }
            df = T.dic_to_df(corr_dict,'pix')
            df = df.dropna(subset=['r'])
            outf = join(outdir,f'{A_var_name}_vs_{B_var_name}_{season}.df')
            T.save_df(df,outf)
            T.df_to_excel(df,outf)


    def phenology_correlation(self,pheno_str,B_data_obj):
        # pheno_str = 'early_start'
        # pheno_str = 'late_end'

        pix_list = DIC_and_TIF().void_spatial_dic()
        Phenology_fpath = join(Phenology().this_class_arr, 'phenology_df/phenology_df.df')
        phenology_df = T.load_df(Phenology_fpath)
        pheno_spatial_dict = {}
        for i, row in tqdm(phenology_df.iterrows(),total=len(phenology_df)):
            pix = row['pix']
            early_start = row[pheno_str]
            early_start_dict = dict(early_start)
            phenology_anomaly_dict = self.phenology_anomaly_dict(early_start_dict)
            phenology_anomaly_list = self.phenology_anomaly_list(phenology_anomaly_dict)
            pheno_spatial_dict[pix] = phenology_anomaly_list
        # A_dict, A_var_name = A_data_obj()
        A_dict = pheno_spatial_dict
        A_var_name = pheno_str
        B_dict, B_var_name = B_data_obj()
        outdir = join(self.this_class_arr, 'seasonal_correlation', f'{A_var_name}_vs_{B_var_name}')
        T.mk_dir(outdir, force=True)
        drought_season_dict = global_drought_season_dict
        for season in drought_season_dict:
            corr_dict = {}
            for pix in tqdm(pix_list, desc=season):
                r_, c_ = pix
                if r_ > 120:
                    continue
                if not pix in A_dict:
                    continue
                if not pix in B_dict:
                    continue
                vals_A_annual = A_dict[pix]
                vals_B = B_dict[pix]
                gs = drought_season_dict[season]
                vals_B_annual = T.monthly_vals_to_annual_val(vals_B, gs)
                try:
                    a, b, r, p = T.nan_line_fit(vals_A_annual, vals_B_annual)
                except:
                    a, b, r, p = np.nan, np.nan, np.nan, np.nan
                corr_dict[pix] = {
                    'a': a,
                    'b': b,
                    'r': r,
                    'p': p,
                }
            df = T.dic_to_df(corr_dict, 'pix')
            df = df.dropna(subset=['r'])
            outf = join(outdir, f'{A_var_name}_vs_{B_var_name}_{season}.df')
            T.save_df(df, outf)
            T.df_to_excel(df, outf)
        pass

    def correlation_tif(self):
        fdir = join(self.this_class_arr,'seasonal_correlation')
        outdir = join(self.this_class_tif,'seasonal_correlation')
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            fdir_i = join(fdir,folder)
            outdir_i = join(outdir,folder)
            T.mk_dir(outdir_i,force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                fpath = join(fdir_i,f)
                outpath = join(outdir_i,f'{f.split(".")[0]}.tif')
                df = T.load_df(fpath)
                spatial_dict_a = T.df_to_spatial_dic(df,'r')
                spatial_dict_p = T.df_to_spatial_dic(df,'p')
                DIC_and_TIF().pix_dic_to_tif(spatial_dict_a,outpath)
                DIC_and_TIF().pix_dic_to_tif(spatial_dict_p,outpath.replace('.tif','_p.tif'))
        # T.open_path_and_file(outdir)

    def plot_correlation_tif(self):
        fdir = join(self.this_class_tif,'seasonal_correlation')
        outdir = join(self.this_class_png,'seasonal_correlation')
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            fdir_i = join(fdir,folder)
            outdir_i = join(outdir,folder)
            T.mk_dir(outdir_i,force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                if f.endswith('_p.tif'):
                    continue
                fpath = join(fdir_i,f)
                print(fpath)
                fpath_p = join(fdir_i,f.replace('.tif','_p.tif'))
                outpath = join(outdir_i,f'{f.split(".")[0]}.png')
                m,ret = Plot().plot_ortho(fpath,vmin=-.8,vmax=.8)
                # m,ret = Plot().plot_ortho(fpath,vmin=-.5,vmax=.5)
                Plot().plot_ortho_significance_scatter(m,fpath_p,temp_root)
                plt.title(f.split('.')[0])
                plt.savefig(outpath,dpi=300)
                plt.close()
                # plt.show()
        T.open_path_and_file(outdir)

    def plot_correlation_bar(self):
        # import statistic
        season_list = ['spring','summer','autumn'][::-1]
        fdir = join(self.this_class_arr, 'seasonal_correlation')
        outdir = join(self.this_class_png, 'plot_correlation_bar')
        ELI_class_list = global_ELI_class_list
        T.mk_dir(outdir, force=True)
        for ELI_class in ELI_class_list:
            plt.figure(figsize=(6, 4))
            for season in season_list:
                for folder in T.listdir(fdir):
                    fdir_i = join(fdir, folder)
                    for f in T.listdir(fdir_i):
                        if not f.endswith('.df'):
                            continue
                        if not season in f:
                            continue
                        fpath = join(fdir_i, f)
                        x_name = f.replace('.df','')
                        df = T.load_df(fpath)
                        # df = statistic.Dataframe_func(df).df
                        # T.save_df(df,fpath)
                        # T.df_to_excel(df,fpath)
                        # T.print_head_n(df)
                        df_ELI = df[df['ELI_class'] == ELI_class]
                        total = len(df_ELI)

                        positive_df = df_ELI[df_ELI['r'] > 0]
                        negative_df = df_ELI[df_ELI['r'] < 0]

                        positive_sig_df = positive_df[positive_df['p'] < 0.05]
                        negative_sig_df = negative_df[negative_df['p'] < 0.05]

                        positive_non_sig_df = positive_df[positive_df['p'] >= 0.05]
                        negative_non_sig_df = negative_df[negative_df['p'] >= 0.05]

                        positive_ratio = len(positive_df) / total
                        negative_ratio = len(negative_df) / total

                        positive_sig_ratio = len(positive_sig_df) / total
                        negative_sig_ratio = len(negative_sig_df) / total
                        print(x_name)
                        x_name = x_name.replace('_vs_NDVI-anomaly_detrend','')

                        plt.barh([x_name,x_name],[positive_ratio,-negative_ratio],color='w',edgecolor='k',zorder=-1)
                        plt.barh([x_name,x_name],[positive_sig_ratio,-negative_sig_ratio],color=['r'],edgecolor='k',zorder=1)
                        plt.text(positive_ratio,x_name,f'{positive_ratio:.2f} ({positive_sig_ratio:.2f})',ha='left',va='center')
                        plt.text(-negative_ratio,x_name,f'{negative_ratio:.2f} ({negative_sig_ratio:.2f})',ha='right',va='center')
            plt.xlim(-2,2)
            plt.title(ELI_class)

            plt.tight_layout()
            outf = join(outdir,f'{ELI_class}.pdf')
            # plt.savefig(outf,dpi=300)
            # plt.close()
            plt.show()
        # T.open_path_and_file(outdir)
        # plt.show()

    def plot_correlation_bar_all_region(self):
        # import statistic
        season_list = ['spring','summer','autumn'][::-1]
        fdir = join(self.this_class_arr, 'seasonal_correlation')
        outdir = join(self.this_class_png, 'plot_correlation_bar')
        T.mk_dir(outdir, force=True)
        plt.figure(figsize=(6, 4))
        for season in season_list:
            for folder in T.listdir(fdir):
                fdir_i = join(fdir, folder)
                for f in T.listdir(fdir_i):
                    if not f.endswith('.df'):
                        continue
                    if not season in f:
                        continue
                    fpath = join(fdir_i, f)
                    x_name = f.replace('.df','')
                    df = T.load_df(fpath)
                    # df = statistic.Dataframe_func(df).df
                    # T.save_df(df,fpath)
                    # T.df_to_excel(df,fpath)
                    # T.print_head_n(df)
                    df_ELI = df
                    total = len(df_ELI)

                    positive_df = df_ELI[df_ELI['r'] > 0]
                    negative_df = df_ELI[df_ELI['r'] < 0]

                    positive_sig_df = positive_df[positive_df['p'] < 0.05]
                    negative_sig_df = negative_df[negative_df['p'] < 0.05]

                    positive_non_sig_df = positive_df[positive_df['p'] >= 0.05]
                    negative_non_sig_df = negative_df[negative_df['p'] >= 0.05]

                    positive_ratio = len(positive_df) / total
                    negative_ratio = len(negative_df) / total

                    positive_sig_ratio = len(positive_sig_df) / total
                    negative_sig_ratio = len(negative_sig_df) / total
                    print(x_name)
                    x_name = x_name.replace('_vs_NDVI-anomaly_detrend','')

                    plt.barh([x_name,x_name],[positive_ratio,-negative_ratio],color='w',edgecolor='k',zorder=-1)
                    plt.barh([x_name,x_name],[positive_sig_ratio,-negative_sig_ratio],color=['r'],edgecolor='k',zorder=1)
                    plt.text(positive_ratio,x_name,f'{positive_ratio:.2f} ({positive_sig_ratio:.2f})',ha='left',va='center')
                    plt.text(-negative_ratio,x_name,f'{negative_ratio:.2f} ({negative_sig_ratio:.2f})',ha='right',va='center')
        plt.xlim(-2,2)
        plt.title('all')

        plt.tight_layout()
        outf = join(outdir,f'all.pdf')
        plt.savefig(outf,dpi=300)
        plt.close()
        # plt.show()
        T.open_path_and_file(outdir)
        # plt.show()

    def correlation_statistic(self):
        outdir = join(self.this_class_arr,'correlation_statistic')
        T.mk_dir(outdir,force=True)

        season_list = ['spring', 'summer', 'autumn'][::-1]
        fdir = join(self.this_class_arr, 'seasonal_correlation')
        ELI_class_list = global_ELI_class_list
        result_dict = {}
        for ELI_class in ELI_class_list:
            for season in season_list:
                for folder in T.listdir(fdir):
                    fdir_i = join(fdir, folder)
                    for f in T.listdir(fdir_i):
                        if not f.endswith('.df'):
                            continue
                        if not season in f:
                            continue
                        fpath = join(fdir_i, f)
                        x_name = f.replace('.df', '')
                        df = T.load_df(fpath)
                        df_ELI = df[df['ELI_class'] == ELI_class]
                        r_mean = np.nanmean(df_ELI['r'])
                        r_std = np.nanstd(df_ELI['r'])
                        key = f'{x_name}_{ELI_class}_{season}'
                        result_dict[key] = {
                            'r_mean':r_mean,
                            'r_std':r_std,
                            'ELI_class':ELI_class,
                            'season':season,
                        }
        df = T.dic_to_df(result_dict,'key')
        outf = join(outdir,'correlation_statistic.df')
        T.df_to_excel(df,outf)
        T.open_path_and_file(outdir)

    def phenology_anomaly_dict(self,phenology_dict):
        vals = list(phenology_dict.values())
        mean = np.nanmean(vals)
        phenology_anomaly_dict = {}
        for year in phenology_dict:
            val = phenology_dict[year]
            anomaly = val - mean
            phenology_anomaly_dict[year] = anomaly
        return phenology_anomaly_dict

    def phenology_anomaly_list(self,phenology_anomaly_dict):
        year_list = sorted(phenology_anomaly_dict.keys())
        phenology_anomaly_list = []
        for year in year_list:
            val = phenology_anomaly_dict[year]
            phenology_anomaly_list.append(val)
        return phenology_anomaly_list


class Optimal_temperature:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Optimal_temperature', result_root_this_script, mode=2)

    def run(self):
        step = .5
        # self.cal_opt_temp(step)
        # self.tif_opt_temp()
        # self.plot_test_cal_opt_temp(step)
        self.resample()
        pass

    def cal_opt_temp(self,step):
        # dff = join(Dataframe_SM().this_class_arr,'dataframe/-0.5.df')
        # df_global = T.load_df(dff)
        # pix_list = T.get_df_unique_val_list(df_global,'pix')

        # step = 1  # Celsius
        # outdir = join(self.this_class_tif,f'optimal_temperature')


        # temp_dic,_ = Load_Data().ERA_Tair_origin()
        temp_dic,_ = Load_Data().Temperature_origin()
        ndvi_dic,vege_name = Load_Data().NDVI_origin()
        # ndvi_dic,vege_name = Load_Data().LT_Baseline_NT_origin()
        # T_dir = join(data_root,'TerraClimate/tmax/per_pix/1982-2020')
        # NDVI_dir = join(data_root,'NDVI4g/per_pix/1982-2020')
        # vege_name = 'NDVI4g'
        # vege_name = 'LT_Baseline_NT'
        outdir = join(self.this_class_arr, f'optimal_temperature',f'{vege_name}_step_{step}_celsius')
        outf = join(outdir,f'{vege_name}_step_{step}_celsius')
        T.mk_dir(outdir,force=True)
        # outdir_i = join(outdir,f'{vege_name}_step_{step}_celsius.tif')
        optimal_temp_dic = {}
        for pix in tqdm(temp_dic):
            if not pix in ndvi_dic:
                continue
            ndvi = ndvi_dic[pix]
            temp = temp_dic[pix]
            temp = np.array(temp)
            temp[temp<0] = np.nan
            if T.is_all_nan(temp):
                continue
            if np.nanstd(temp) == 0:
                continue
            # temp = np.array(temp) - 273.15  # Kelvin to Celsius
            df = pd.DataFrame()
            df['ndvi'] = ndvi
            df['temp'] = temp
            df = df[df['ndvi'] > 0]
            df = df.dropna()
            if len(df) == 0:
                continue
            max_t = max(df['temp'])
            min_t = int(min(df['temp']))
            t_bins = np.arange(start=min_t, stop=max_t, step=step)
            df_group, bins_list_str = T.df_bin(df, 'temp', t_bins)
            quantial_90_list = []
            x_list = []
            for name, df_group_i in df_group:
                vals = df_group_i['ndvi'].tolist()
                if len(vals) == 0:
                    continue
                quantile_90 = np.nanquantile(vals, 0.9)
                left = name[0].left
                x_list.append(left)
                quantial_90_list.append(quantile_90)
            # multi regression to find the optimal temperature
            # parabola fitting
            x = np.array(x_list)
            y = np.array(quantial_90_list)
            if len(x) < 3:
                continue
            if len(y) < 3:
                continue
            a, b, c = self.nan_parabola_fit(x, y)
            y_fit = a * x ** 2 + b * x + c
            T_opt = x[np.argmax(y_fit)]
            # print(T_opt)
            # exit()
            optimal_temp_dic[pix] = T_opt
        T.save_npy(optimal_temp_dic, outf)

    def tif_opt_temp(self):
        outdir = join(self.this_class_tif,f'optimal_temperature')
        T.mk_dir(outdir,force=True)
        fpath = join(self.this_class_arr,'optimal_temperature/NDVI-origin_step_0.5_celsius/NDVI-origin_step_0.5_celsius.npy')
        spatial_dict = T.load_npy(fpath)
        spatial_dict_new = {}
        for pix in spatial_dict:
            val = spatial_dict[pix]
            # val = val + 273.15
            spatial_dict_new[pix] = val
        outf = join(outdir,f'optimal_temperature.tif')
        DIC_and_TIF().pix_dic_to_tif(spatial_dict_new,outf)

    def kernel_cal_opt_temp(self,params):
        NDVI_dir,T_dir,outdir,step,f,pix_list = params
        fpath_NDVI = join(NDVI_dir, f)
        fpath_T = join(T_dir, f)
        ndvi_dic = T.load_npy(fpath_NDVI)
        temp_dic = T.load_npy(fpath_T)
        optimal_temp_dic = {}
        for pix in pix_list:
            if not pix in ndvi_dic:
                continue
            ndvi = ndvi_dic[pix]
            temp = temp_dic[pix]
            temp = np.array(temp)
            # temp = np.array(temp) - 273.15  # Kelvin to Celsius
            df = pd.DataFrame()
            df['ndvi'] = ndvi
            df['temp'] = temp
            df = df[df['ndvi'] > 0]
            df = df.dropna()
            if len(df) == 0:
                continue
            max_t = max(df['temp'])
            min_t = int(min(df['temp']))
            t_bins = np.arange(start=min_t, stop=max_t, step=step)
            df_group, bins_list_str = T.df_bin(df, 'temp', t_bins)
            quantial_90_list = []
            x_list = []
            for name, df_group_i in df_group:
                vals = df_group_i['ndvi'].tolist()
                if len(vals) == 0:
                    continue
                quantile_90 = np.nanquantile(vals, 0.9)
                left = name[0].left
                x_list.append(left)
                quantial_90_list.append(quantile_90)
            # multi regression to find the optimal temperature
            # parabola fitting
            x = np.array(x_list)
            y = np.array(quantial_90_list)
            if len(x) < 3:
                continue
            if len(y) < 3:
                continue
            a, b, c = self.nan_parabola_fit(x, y)
            y_fit = a * x ** 2 + b * x + c
            T_opt = x[np.argmax(y_fit)]
            optimal_temp_dic[pix] = T_opt
        outf = join(outdir, f)
        T.save_npy(optimal_temp_dic, outf)

    def plot_test_cal_opt_temp(self,step):

        # step = 1  # Celsius

        temp_dic,_ = Load_Data().Temperature_origin()
        ndvi_dic,_ = Load_Data().NDVI_origin()
        # ndvi_dic,_ = Load_Data().LT_Baseline_NT_origin()
        optimal_temp_dic = {}
        for pix in tqdm(temp_dic):
            ndvi = ndvi_dic[pix]
            temp = temp_dic[pix]
            ndvi = np.array(ndvi)
            temp = np.array(temp)
            temp[temp < 0] = np.nan
            if T.is_all_nan(temp):
                continue
            if np.nanstd(temp) == 0:
                continue
            if np.nanstd(ndvi) == 0:
                continue
            ndvi[ndvi < 0] = np.nan
            ndvi[ndvi>10000] = np.nan
            if True in np.isnan(ndvi):
                continue
            # if T.is_all_nan(ndvi):
            #     continue
            # print(temp)
            # print(ndvi)
            # exit()
            df = pd.DataFrame()
            df['ndvi'] = ndvi
            df['temp'] = temp
            df = df[df['ndvi'] > 0.1]
            df = df.dropna()
            max_t = max(df['temp'])
            min_t = int(min(df['temp']))
            t_bins = np.arange(start=min_t,stop=max_t,step=step)
            df_group, bins_list_str = T.df_bin(df,'temp',t_bins)
            # ndvi_list = []
            # box_list = []
            color_list = T.gen_colors(len(df_group))
            color_list = color_list[::-1]
            flag = 0
            quantial_90_list = []
            x_list = []
            for name,df_group_i in df_group:
                vals = df_group_i['ndvi'].tolist()
                quantile_90 = np.nanquantile(vals,0.9)
                left = name[0].left
                x_list.append(left)
                # plt.scatter([left]*len(vals),vals,s=20,color=color_list[flag])
                flag += 1
                quantial_90_list.append(quantile_90)
            x = np.array(x_list)
            y = np.array(quantial_90_list)
            a,b,c = self.nan_parabola_fit(x,y)
            y_fit = a*x**2 + b*x + c
            # plt.plot(x,y_fit,'k--',lw=2)
            opt_T = x[np.argmax(y_fit)]
            if opt_T < 20:
                continue
            if np.argmax(y_fit) >= 39:
                continue

            #######################
            flag = 0
            quantial_90_list = []
            x_list = []
            for name, df_group_i in df_group:
                vals = df_group_i['ndvi'].tolist()
                quantile_90 = np.nanquantile(vals, 0.9)
                left = name[0].left
                x_list.append(left)
                plt.scatter([left] * len(vals), vals, s=20, color=color_list[flag])
                flag += 1
                quantial_90_list.append(quantile_90)
            x = np.array(x_list)
            # print(len(x))
            # exit()
            y = np.array(quantial_90_list)
            a, b, c = self.nan_parabola_fit(x, y)
            y_fit = a * x ** 2 + b * x + c
            plt.plot(x, y_fit, 'k--', lw=2)
            opt_T = x[np.argmax(y_fit)]
            #######################
            print('opt_T',opt_T)
            print('argmax',np.argmax(y_fit))
            plt.scatter([opt_T],[np.max(y_fit)],s=200,marker='*',color='r',zorder=99)
            # print(len(y_fit))
            # print(len(quantial_90_list))
            a_,b_,r_,p_ = T.nan_line_fit(y_fit,quantial_90_list)
            r2 = r_**2
            # print(r2)
            # exit()


            plt.plot(x_list,quantial_90_list,c='k',lw=2)
            plt.title(f'a={a:.3f},b={b:.3f},c={c:.3f}')
            # print(t_bins)
            # # plt.plot(t_bins[:-1],ndvi_list)
            # plt.boxplot(box_list,positions=t_bins[:-1],showfliers=False)
            plt.show()



    def nan_parabola_fit(self, val1_list, val2_list):
        if not len(val1_list) == len(val2_list):
            raise UserWarning('val1_list and val2_list must have the same length')
        val1_list_new = []
        val2_list_new = []
        for i in range(len(val1_list)):
            val1 = val1_list[i]
            val2 = val2_list[i]
            if np.isnan(val1):
                continue
            if np.isnan(val2):
                continue
            val1_list_new.append(val1)
            val2_list_new.append(val2)
        a,b,c = np.polyfit(val1_list_new, val2_list_new, 2)

        return a,b,c

    def resample(self):
        fpath = '/Volumes/NVME2T/Energy_water_hotdrought/results/analysis/Optimal_temperature/tif/optimal_temperature/LT_Baseline_NT_origin_step_0.5_celsius.tif'
        outpath = '/Volumes/NVME2T/Energy_water_hotdrought/results/analysis/Optimal_temperature/tif/optimal_temperature/LT_Baseline_NT_origin_step_0.5_celsius_resample.tif'
        ToRaster().resample_reproj(fpath, outpath, 0.5)
        pass

def line_to_shp(inputlist, outSHPfn):
    ############重要#################
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    ############重要#################
    # start,end,outSHPfn,val1,val2,val3,val4,val5
    # _,_,_,_=start[1],start[0],end[0],end[1]

    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbLineString)

    # create line geometry
    line = ogr.Geometry(ogr.wkbLineString)

    for i in range(len(inputlist)):
        start = inputlist[i][0]
        end = inputlist[i][1]

        line.AddPoint(start[0], start[1])
        line.AddPoint(end[0], end[1])

        featureDefn = outLayer.GetLayerDefn()
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(line)
        outLayer.CreateFeature(outFeature)
        outFeature.Destroy()
        line = ogr.Geometry(ogr.wkbLineString)
        outFeature = None

    # define the spatial reference, WGS84
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromEPSG(4326)
    spatialRef.MorphToESRI()
    file = open(outSHPfn[:-4] + '.prj', 'w')
    file.write(spatialRef.ExportToWkt())
    file.close()


def gen_world_grid_shp():
    x_interval = 30
    y_interval = 30

    outf = join(this_root, 'shp/world_line/world_grid.shp')
    lon_list = np.arange(-180, 180, x_interval)
    lat_list = np.arange(-90, 90, y_interval)
    lon_list = np.array(lon_list, dtype=float)
    lat_list = np.array(lat_list, dtype=float)
    # print(lon_list)
    # exit()
    lon_lines = []
    lat_lines = []
    for lon in lon_list:
        lon_lines.append([[lon, -90], [lon, 90]])
    for lat in lat_list:
        lat_lines.append([[-180, lat], [180, lat]])
    lines = lon_lines + lat_lines
    line_to_shp(lines, outf)

    pass


def main():
    # Water_energy_limited_area().run()
    # Water_energy_limited_area_daily().run()
    # Max_Scale_and_Lag_correlation_SPEI().run()
    Pick_Drought_Events().run()
    # Pick_Drought_Events_SM().run()
    # Resistance_Resilience().run()
    # Net_effect_annual().run()
    # Net_effect_monthly().run()
    # Phenology().run()
    # Long_term_correlation().run()
    Optimal_temperature().run()

    # gen_world_grid_shp()
    pass


if __name__ == '__main__':
    main()