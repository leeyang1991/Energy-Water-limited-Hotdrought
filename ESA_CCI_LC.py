# coding=utf-8
from lytools import *
from pprint import pprint
import xarray as xr
T = Tools()

class ESA_CCI:
    def __init__(self):
        # self.data_dir = r"F:\yang\ESA-CCI_LC"
        # self.data_dir = '/mnt/disk2/yang/ESA-CCI_LC'
        self.data_dir = '/mnt/disk3/ESA_CCI_LC/'
        pass

    def run(self):
        self.aggregate_lc()
        pass


    def return_variables_list(self):
        variable_list_str = '''WATER
        BARE
        BUILT
        GRASS-MAN
        GRASS-NAT
        SHRUBS-BD
        SHRUBS-BE
        SHRUBS-ND
        SHRUBS-NE
        WATER_INLAND
        SNOWICE
        TREES-BD
        TREES-BE
        TREES-ND
        TREES-NE
        LAND'''
        variables_list = variable_list_str.replace(' ','')
        variables_list = variables_list.split('\n')
        return variables_list

    def aggregate_lc(self):
        fdir = join(self.data_dir,'ESACCI-LC-L4-PFT-Map-300m-P1Y-1992-2020-v2.0.8')
        outdir = join(self.data_dir,'tif')
        T.mk_dir(outdir,force=True)
        params_list = []
        for f in T.listdir(fdir):
            params = [fdir,f,outdir]
            params_list.append(params)
        MULTIPROCESS(self.kernel_aggregate_lc,params_list).run(process=24)

    def kernel_aggregate_lc(self,params):
        fdir,f,outdir = params
        fpath = join(fdir, f)
        nc_r = xr.open_dataset(fpath)
        variables_list = self.return_variables_list()
        # agb_arr = nc_r['agb']
        lon = nc_r['lon']
        lat = nc_r['lat']
        # print(len(lon),len(lat)) # 129600 64800
        pixelWidth = 360 / len(lon)
        pixelHeight = 180 / len(lat)
        new_pixelWidth = 0.5
        new_pixelHeight = 0.5
        nx = new_pixelWidth / pixelWidth
        ny = new_pixelHeight / pixelHeight

        new_lon = np.arange(-180, 180, new_pixelWidth)
        new_lat = np.arange(90, -90, -new_pixelHeight)
        spatial_dict = {}
        for new_r in tqdm(range(len(new_lat))):
            c_range = [new_r * nx, (new_r + 1) * nx]
            c_range_int = [int(c_range[0]), int(c_range[1])]
            for new_c in range(len(new_lon)):
                r_range = [new_c * ny, (new_c + 1) * ny]
                r_range_int = [int(r_range[0]), int(r_range[1])]
                pix = (new_r, new_c)
                spatial_dict_i = {}
                for var in variables_list:
                    # spatial_dict[(new_r, new_c)] = nc_r[var][r_range[0]:r_range[1], c_range[0]:c_range[1]]
                    # print(vals)
                    try:
                        vals = nc_r[var][0][r_range_int[0]:r_range_int[1], c_range_int[0]:c_range_int[1]]
                        vals_values = vals.values
                        vals_values = np.array(vals_values, dtype=np.float32)
                        vals_mean = np.nanmean(vals_values)
                    except:
                        vals_mean = np.nan
                    spatial_dict_i[var] = vals_mean
                spatial_dict[pix] = spatial_dict_i
        df_result = T.dic_to_df(spatial_dict, 'pix')
        outf = join(outdir, f + '.df')
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)
        pass


def main():
    ESA_CCI().run()

    pass

if __name__ == '__main__':
    main()