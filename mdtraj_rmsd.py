import os
from os.path import isfile
import sys
from glob import glob
import re
from itertools import islice
from typing import List
import mdtraj as md
import math
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from Bio import PDB
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import manifold
from collections import defaultdict
import time
from time import sleep


class PCA_rmsd():
    def Bfactor(self, pdb: str) -> List:
        """
        take in a pdb file and identify the index of every alpha carbon
        """
        structure = PDB.PDBParser(QUIET=True).get_structure('protein', pdb)
    
        _bfactor = []
    
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        _bfactor.append(residue['CA'].bfactor)
        return _bfactor
    
    
    def Alpha_Carbon_Indices(self, pdb: str) -> List:
        """
        take in a pdb file and identify the index of every alpha carbon
        """
        structure = PDB.PDBParser(QUIET=True).get_structure('protein', pdb)
    
        alpha_carbons = []
    
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        resid = residue.resname
                        alpha_carbons.append([resid, residue['CA'].get_serial_number() - 1])
        return alpha_carbons
    
    def Match_Alpha_Carbons(self, pdb_1: str, pdb_2: str) -> List[int]:
        """
        Take in two pdb structure files and search through them for matching alpha carbons
        This should identify positions correctly even if sequences are not identical
        """
        alpha_c_1 = self.Alpha_Carbon_Indices(pdb_1)
        alpha_c_2 = self.Alpha_Carbon_Indices(pdb_2)
    
        matching_alpha_carbons1 = []
        matching_alpha_carbons2 = []
    
        for i, (resname_1, ca_index1) in enumerate(alpha_c_1):
            for j, (resname_2, ca_index2) in enumerate(alpha_c_2):
                if resname_2 == resname_1 and ca_index1 not in [_[1] for _ in matching_alpha_carbons1] and ca_index2 not in [_[1] for _ in matching_alpha_carbons2]:
                    #prevent erroneous match at NTD
                    if i > 0 and j > 0:
                        if alpha_c_1[i-1][0] != alpha_c_2[j-1][0]: #check previous matches
                            continue
                    # prevent erroneous backtracking
                    if len(matching_alpha_carbons1) > 2 and len(matching_alpha_carbons2) > 2:
                        if ca_index2 < matching_alpha_carbons2[-1][-1]:
                            continue
                    #prevent erroneous match at CTD
                    if i < len(alpha_c_1) - 1 and j < len(alpha_c_2) - 1:
                        if alpha_c_1[i+1][0] != alpha_c_2[j+1][0]: #check next matches
                            continue
                    matching_alpha_carbons1.append([resname_1, ca_index1])
                    matching_alpha_carbons2.append([resname_2, ca_index2])
                    break
        #skip first residue to avoid erroneous glycine match
        return matching_alpha_carbons1[1:], matching_alpha_carbons2[1:]
    
    def Calculate_RMSD(self, structure_1: str, structure_2: str, structure_1_index: List[int], structure_2_index: List[int]) -> int:
        """
        calculate the RMSD between two structures using MDtraj library
        this script will fail if mdtraj is not loaded in your python environment
        recommend python 3.10
        """
    
        #load structure information in mdtraj
        pdb = md.load(structure_1)
        pdb_ca = pdb.atom_slice(structure_1_index) #select only CA atoms
    
        #load structure information in mdtraj
        reference = md.load(structure_2)
        reference_ca = reference.atom_slice(structure_2_index) #select only CA atoms
    
        # Calculate RMSD of CA atoms
        pdb_ca.superpose(reference_ca)
        return md.rmsd(pdb_ca, reference_ca, frame=0)
    
    
    #def Parse_Foldseek(self, path: str):
    def Parse_Foldseek(self, path: str):
        foldseek_dict = defaultdict(list)
        parse_dict = {"query" : 0,"target" : 1,"alntmscore" : 2,"qaln" : 3,"taln" : 4,"alnlen" : 5,"evalue" : 6,"bits" : 7}
        if os.path.getsize(path) > 0:
            with open(path, 'r') as file:
                for line in file:
                    _ = re.split(r'\s+', line.strip())
                    foldseek_dict['Path'].append(path)
                    for key, data in parse_dict.items():
                        foldseek_dict[key].append(_[data])
        else:
            foldseek_dict['Path'].append(path)
            foldseek_dict['query'].append('NA')
            foldseek_dict['target'].append('Undefined')
            foldseek_dict['alntmscore'].append(np.nan)
            foldseek_dict['qaln'].append(np.nan)
            foldseek_dict['taln'].append(np.nan)
            foldseek_dict['alnlen'].append(np.nan)
            foldseek_dict['evalue'].append(np.nan)
            foldseek_dict['bits'].append(0)
        return foldseek_dict
    
    #def CL_input():
    #    """
    #    Parse command line arguments that are being passed in
    #    """
    #    if not any((True if _ == '-path' else False for _ in sys.argv)) or len(sys.argv) < 3:
    #        print('Missing command line arguments!')
    #        print('Available flags:')
    #        print("-path ####  |  Directory containing CF-random runs")
    #        print("-ref  ####  |  Optional Directory containing reference structures")
    #        sys.exit()
    #
    #    path = sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-path' == _][0] + 1]
    #    ref  = None if not [idx for idx, _ in enumerate(sys.argv) if '-ref' == _] else sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-ref' == _][0] + 1]
    #    return path, ref




    def __init__(self, pdb1_name, blind_path):
        """
        This script requires that both colabfold and Phenix have been run.
        The directories should have .pdb structures from colabfold.
        The Phenix files should have the same naming convention as colabfold but end with .mol
        """

        #path, ref = CL_input()
        path = blind_path
        #ref = None if not [idx for idx, _ in enumerate(sys.argv) if '-ref' == _] else sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-ref' == _][0] + 1]



        pdb_structures = sorted(glob(path + '/*/0_unrelaxed_*.pdb'))
        # mol_files = sorted(glob(path + '/*/*r3.mol'))
        foldseek_files = sorted(glob(path + '/*/0_unrelaxed_*.foldseek'))


        #___FOLDSEEK_SECTION____________________________________________________________`o____________________________________
        depth = []
        for cate in ['full', '16_32', '1_2', '2_4', '32_64', '4_8', '64_128', '8_16']:
            for deta in range(0, 25):
                depth = np.append(depth, cate)
        depth = depth.tolist()


        df_phenix = defaultdict(list)
        for foldseek in foldseek_files:
            _ = self.Parse_Foldseek(foldseek)
            for key, data in _.items():
                df_phenix[key].append(data[0])
        df_phenix['file name'] = df_phenix['query']


        df_phenix['depth'] = depth

        #___FOLDSEEK_SECTION________________________________________________________________________________________________
        def batched(iterable, n):
            # batched('ABCDEFG', 3) → ABC DEF G
            if n < 1:
                raise ValueError('n must be at least one')
            it = iter(iterable)
            while batch := tuple(islice(it, n)):
                yield batch

        #___RMSD_SECTION________________________________________________________________________________________________

        #all colabfold structures have the same sequence
        list_CA_index_pdb1, list_CA_index_pdb2 = self.Match_Alpha_Carbons(pdb_structures[0], pdb_structures[3])
        #loop through and find window that returns largest tmscore difference!!!
        CAs = [_[1] for _ in list_CA_index_pdb1]

        #if os.path.isfile(path.split('/')[0] + '_rmsd.npy'):
        if os.path.isfile(pdb1_name + '_rmsd.npy'):
            rmsd_array = np.load(pdb1_name + '_rmsd.npy')
            #rmsd_array = np.load(path.split('/')[0] + '_rmsd.npy')
            bfactor = []
            for idx_base, pdb_base in enumerate(pdb_structures):
                bfactor.append(np.average(self.Bfactor(pdb_base)))

        else:
            print('RMSD array calculation')
            filter_rmsd_windows = 5; window_step = 4;bfactor = [] # remove all windows generated by batched that have fewer than 5 atoms; step by ~10% protein length
            batches = list(batched(CAs, n=len(CAs) // window_step))
            batches = [tuple(CAs) , *batches]
            rmsd_array = np.empty([len(pdb_structures), len(pdb_structures)*len([b for b in batches if len(b) >= filter_rmsd_windows])], dtype='f')

            #NOTE Calculate_RMSD : path_to_pdb1, path_to_pdb2, list_of_CA_indices_for_pdb1, list_of_CA_indices_for_pdb1,
            for idx_base, pdb_base in enumerate(pdb_structures):
                bfactor.append(np.average(self.Bfactor(pdb_base)))
                rmsd_vec = np.array([], dtype='f')
                
                for idx, pdb in enumerate(pdb_structures):
                    if (idx % 4) == 0:
                        sys.stdout.write('\r')
                        # the exact output you're looking for
                        sys.stdout.write("[%-50s] %d%%" % ('='*int(idx / 4), 0.5*idx))
                        sys.stdout.flush()
                        sleep(0.25)
                    #else:
                    #    sys.stdout.write('\r')
                    #    # the exact output you're looking for
                    #    sys.stdout.write("[%-50s] %d%%" % ('='*idx, 0.5*idx))
                    #    sys.stdout.flush()
                    #    sleep(0.25)

                    #print('_'*idx + '|' + '-'*(len(pdb_structures)-idx), "{:.2f}%".format(idx_base / len(pdb_structures)*100), end='\r', flush=True)
                    _ = np.array([self.Calculate_RMSD(pdb_base, pdb, b,b) for b in batches if len(b) >= filter_rmsd_windows], dtype = 'f')
                    rmsd_vec = np.append(rmsd_vec, _)
                    pad = len(_) #this should always be run first becuase idx_base and idx start at 0,0
                rmsd_array[idx_base] = rmsd_vec

            #with open(path.split('/')[0] + '_rmsd.npy', 'wb') as f:
            with open(pdb1_name + '_rmsd.npy', 'wb') as f:
                np.save(f, rmsd_array)

        #___RMSD_SECTION________________________________________________________________________________________________
        #___Dimensionality Reduction_SECTION________________________________________________________________________________________________

        # z-score normalize
        rmsd_array = rmsd_array - np.mean(rmsd_array, axis=0) # mean-centering
        rmsd_array = rmsd_array / np.std(rmsd_array, axis=0)  # scaling to have sd=1

        sklearn_pca = sklearnPCA(n_components=3)
        rmsd_pca = sklearn_pca.fit_transform(rmsd_array)


        seed = 1
        sklearn_mds = manifold.MDS(
                        n_components=2,
                        metric=True,
                        max_iter=3000,
                        eps=1e-9,
                        random_state=seed,
                        n_jobs=1,
                        n_init=4)
        rmsd_mds = sklearn_mds.fit_transform(rmsd_array)
        #___Dimensionality Reduction_SECTION________________________________________________________________________________________________

        df_phenix['RMSD PC 1'] = rmsd_pca[:,0]
        df_phenix['RMSD PC 2'] = rmsd_pca[:,1]
        df_phenix['RMSD MDS 1'] = rmsd_mds[:,0]
        df_phenix['RMSD MDS 2'] = rmsd_mds[:,1]

        df_phenix['bfactor'] = bfactor
        for key in df_phenix:
            print(key, len(df_phenix[key]))
        df_phenix = pd.DataFrame.from_dict(df_phenix)
        df_phenix = df_phenix.sort_values(by=['RMSD PC 1', 'RMSD PC 2'])
        #df_phenix.to_csv(path.split('/')[0] + '.csv', index=False)
        df_phenix.to_csv(pdb1_name + '.csv', index=False)
        #___Contour-map_SECTION________________________________________________________________________________________________
        

        df_phenix['bits'] = df_phenix['bits'].astype(int)
        bits_max = df_phenix['bits'].max()
        df_phenix = df_phenix[df_phenix.bits > float(bits_max * 0.65)]


        x_full_ave_tmp = np.average(df_phenix.loc[df_phenix['depth'].eq('full'), 'RMSD PC 1'])
        y_full_ave_tmp = np.average(df_phenix.loc[df_phenix['depth'].eq('full'), 'RMSD PC 2'])

        x_full_ave = np.average(x_full_ave_tmp); y_full_ave = (y_full_ave_tmp)

        clus_dist_diff = []


        def is_nan(value):
            return math.isnan(float(value))


        dp_list = ['16_32', '1_2', '2_4', '32_64', '4_8', '64_128', '8_16'] 
        for dp in dp_list:
            tmp_clus_dist_diff = []
            x_var_ave = []; y_var_ave = []; x_var_ave_tmp = []; y_var_ave_tmp = [];
            
            x_var_ave_tmp = np.average(df_phenix.loc[df_phenix['depth'].eq(dp), 'RMSD PC 1'])
            y_var_ave_tmp = np.average(df_phenix.loc[df_phenix['depth'].eq(dp), 'RMSD PC 2'])

            if df_phenix.loc[df_phenix['depth'].eq(dp), 'RMSD PC 1'].empty:
                print("Dataframe is empty")
                x_var_ave_tmp = 0; y_var_ave_tmp = 0;
                tmp_clus_dist_diff = 0
                clus_dist_diff.append(tmp_clus_dist_diff)
                print(tmp_clus_dist_diff)
            else:
                x_var_ave = np.average(x_var_ave_tmp); y_var_ave = np.average(y_var_ave_tmp)

                tmp_clus_dist_diff = float(np.sqrt((x_full_ave - x_var_ave)**2 + (y_full_ave - y_var_ave)**2))
                if tmp_clus_dist_diff < 25:
                    tmp_clus_dist_diff = 0
                    clus_dist_diff.append(tmp_clus_dist_diff)
                else:
                    clus_dist_diff.append(tmp_clus_dist_diff)


        print(clus_dist_diff) 

        depth_sel = ['full', dp_list[np.argmax(clus_dist_diff)]]
        print(depth_sel)






        def gauss(x_grid, y_grid, x, y, z):
            s = 1.5
            return (np.sqrt(2 * np.pi * s)) * np.exp(-(((x - x_grid)/s)**2 + ((y - y_grid)/s)**2))


        f, ax = plt.subplots(1,1,figsize=(10,7)) 
        plt.xlabel('RMSD PC 1', fontsize=15); plt.ylabel('RMSD PC 2', fontsize=15)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.80, box.height* 0.80])


        clus_dist_diff = np.array(clus_dist_diff)


        if np.all(clus_dist_diff == 0):
            print("negative control")
    
            #df_phenix = df_phenix[df_phenix.bits > float(bits_max * 0.65)]
    
    
            x_adjust = abs(min((df_phenix['RMSD PC 1'].min() * 0.25, df_phenix['RMSD PC 1'].max() * 0.25)))
            y_adjust = abs(min((df_phenix['RMSD PC 2'].min() * 0.25, df_phenix['RMSD PC 2'].max() * 0.25)))
            x_min = int(df_phenix['RMSD PC 1'].min() - x_adjust); x_max = int(df_phenix['RMSD PC 1'].max() + x_adjust)
            y_min = int(df_phenix['RMSD PC 2'].min() - y_adjust); y_max = int(df_phenix['RMSD PC 2'].max() + y_adjust)
    
            z = df_phenix['bits'].to_numpy()
            x = df_phenix['RMSD PC 1'].to_numpy()
            y = df_phenix['RMSD PC 2'].to_numpy()
            z = z.astype('f'); x = x.astype('f'); y = y.astype('f')
            x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 2000), np.linspace(y_min,y_max, 2000))
            z_grid = np.zeros(x_grid.shape)
            # vmax = np.max(z); vmin = np.min(z)
            #vmax excludes bit scores from the full msa
            vmax = df_phenix[df_phenix['Path'].map(lambda x:'Full' not in  x) == True]['bits'].map(lambda x: int(x)).max()
    
    
    
    
            tree = cKDTree(list(zip(x,y)))
            print("tree result")
            for (_x, _y, _z) in zip(x, y, z):
                _ = gauss(x_grid, y_grid, _x, _y, _z)  #generate gaussian
                _ = (_ / np.max(_))
                # _[_ < 0.5] = 0                         #makes the plot look more bubbly
                #scale by current value of z
                _ = _ * float(_z) if _z <= vmax else _ * float(vmax)
                #dist, idx = tree.query((_x,_y), k=4)
                dist, idx = tree.query((_x,_y), k=5)
                # reduce intensity of isolated configurations
                if sum(i > 2.0 for i in dist) > 1:
                #if sum(i > 2.0 for i in dist) > 0.5:
                    _ = _ * 1.0
                z_grid = np.maximum(z_grid, _)         #don't allow gaussians to build
    
    
            #for col_size in range(0, np.size(x_grid, 0)):
            #    if np.any(z_grid[col_size, :] > 100):
            #        plt.scatter(x_grid[col_size, :], y_grid[col_size, :], s=35, marker="o")
    
            contour = plt.contourf(x_grid, y_grid, z_grid, levels = 100, cmap='Greys')
    
    
            cmap = cm._colormaps['rainbow_r']
            norm = plt.Normalize(0, 100)
            axes = []; axes_depth = []
            #each 'type' needs its own handle for matplotlib to give unique legend elements
            color_dict = defaultdict(int)
            for idx, t in enumerate(df_phenix['target'].unique()):
                color_dict[t] = 100 * (idx / len(df_phenix['target'].unique()))
            df_phenix['target_color'] = df_phenix['target'].map(lambda x: color_dict[x])
    
    
            for idx, t in enumerate(df_phenix['target'].unique()):
                color_dict[t] = 100 * (idx / len(df_phenix['target'].unique()))
            df_phenix['target_color'] = df_phenix['target'].map(lambda x: color_dict[x])
    
    
            for t in df_phenix['target'].unique():
                axes.append(ax.scatter(x=df_phenix.loc[df_phenix['target'].eq(t), 'RMSD PC 1'],
                                       y=df_phenix.loc[df_phenix['target'].eq(t), 'RMSD PC 2'],
                                       c=df_phenix.loc[df_phenix['target'].eq(t), 'target_color'].map(lambda x: cm.colors.rgb2hex(cmap(norm(x)), keep_alpha=True)),
                                       s=80, linewidth=0, linestyle="None"))
    
    
            plt.legend(df_phenix['target'].unique(),loc='center left', bbox_to_anchor=(1.25, 0.5))
            plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    
            plt.colorbar(contour, label = 'Bit Score')
            plt.tight_layout()
    
            #plt.savefig(path.split('/')[0] + 'RMSD_PC.png')
            plt.savefig(pdb1_name + '_RMSD_PC.png')




        else:
            ## count the number of protein is single or not
            target_count = df_phenix['target'].unique()
            total_target_count = len(df_phenix['target'].unique())
            print(total_target_count)
            print("slicing test")
            target_count = ''.join(target_count)
            xxx = str(target_count)
            yyy = xxx.split('-')[0]
            print(yyy); print(pdb1_name[:4])


            print("fold-switching or alternative conformation")
    
            df_phenix_full = df_phenix[df_phenix.depth.eq('full')]
            df_phenix_var  = df_phenix[df_phenix.depth.eq(dp_list[np.argmax(clus_dist_diff)])]
            df_phenix = pd.concat([df_phenix_full, df_phenix_var], axis=0)
    
            x_adjust = abs(min((df_phenix['RMSD PC 1'].min() * 0.25, df_phenix['RMSD PC 1'].max() * 0.25)))
            y_adjust = abs(min((df_phenix['RMSD PC 2'].min() * 0.25, df_phenix['RMSD PC 2'].max() * 0.25)))
            x_min = int(df_phenix['RMSD PC 1'].min() - x_adjust); x_max = int(df_phenix['RMSD PC 1'].max() + x_adjust)
            y_min = int(df_phenix['RMSD PC 2'].min() - y_adjust); y_max = int(df_phenix['RMSD PC 2'].max() + y_adjust)
    
            z = df_phenix['bits'].to_numpy()
            x = df_phenix['RMSD PC 1'].to_numpy()
            y = df_phenix['RMSD PC 2'].to_numpy()
            z = z.astype('f'); x = x.astype('f'); y = y.astype('f')
            print(x, y, z)
            x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 2000), np.linspace(y_min,y_max, 2000))
            z_grid = np.zeros(x_grid.shape)
            # vmax = np.max(z); vmin = np.min(z)
            #vmax excludes bit scores from the full msa
            vmax = df_phenix[df_phenix['Path'].map(lambda x:'Full' not in  x) == True]['bits'].map(lambda x: int(x)).max()
    
            tree = cKDTree(list(zip(x,y)))
            print("tree result")
            print(tree)
            for (_x, _y, _z) in zip(x, y, z):
                _ = gauss(x_grid, y_grid, _x, _y, _z)  #generate gaussian
                _ = (_ / np.max(_))
                # _[_ < 0.5] = 0                         #makes the plot look more bubbly
                #scale by current value of z
                _ = _ * float(_z) if _z <= vmax else _ * float(vmax)
                #dist, idx = tree.query((_x,_y), k=4)
                dist, idx = tree.query((_x,_y), k=5)
                # reduce intensity of isolated configurations
                if sum(i > 2.0 for i in dist) > 1:
                #if sum(i > 2.0 for i in dist) > 0.5:
                    _ = _ * 0.75
                z_grid = np.maximum(z_grid, _)         #don't allow gaussians to build
    
    
            #for col_size in range(0, np.size(x_grid, 0)):
            #    if np.any(z_grid[col_size, :] > 100):
            #        plt.scatter(x_grid[col_size, :], y_grid[col_size, :], s=35, marker="o")
    
            contour = plt.contourf(x_grid, y_grid, z_grid, levels = 100, cmap='Greys')
    
    
            cmap = cm._colormaps['rainbow_r']
            norm = plt.Normalize(0, 100)
            axes = []; axes_depth = []
            #each 'type' needs its own handle for matplotlib to give unique legend elements
            color_dict = defaultdict(int)
            for idx, t in enumerate(df_phenix['target'].unique()):
                color_dict[t] = 100 * (idx / len(df_phenix['target'].unique()))
            df_phenix['target_color'] = df_phenix['target'].map(lambda x: color_dict[x])
    
    
            for idx, t in enumerate(df_phenix['target'].unique()):
                color_dict[t] = 100 * (idx / len(df_phenix['target'].unique()))
            df_phenix['target_color'] = df_phenix['target'].map(lambda x: color_dict[x])
    
    
            for t in df_phenix['target'].unique():
                axes.append(ax.scatter(x=df_phenix.loc[df_phenix['target'].eq(t), 'RMSD PC 1'],
                                       y=df_phenix.loc[df_phenix['target'].eq(t), 'RMSD PC 2'],
                                       c=df_phenix.loc[df_phenix['target'].eq(t), 'target_color'].map(lambda x: cm.colors.rgb2hex(cmap(norm(x)), keep_alpha=True)),
                                       s=80, linewidth=0, linestyle="None"))
    
            for t in df_phenix['target'].unique():
                axes.append(ax.scatter(x=df_phenix.loc[df_phenix['target'].eq(t), 'RMSD PC 1'],
                                       y=df_phenix.loc[df_phenix['target'].eq(t), 'RMSD PC 2'],
                                       c=df_phenix.loc[df_phenix['target'].eq(t), 'target_color'].map(lambda x: cm.colors.rgb2hex(cmap(norm(x)), keep_alpha=True)),
                                       s=80, linewidth=0, linestyle="None"))
    
            plt.legend(df_phenix['target'].unique(),loc='center left', bbox_to_anchor=(1.25, 0.5))
            plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    
            plt.colorbar(contour, label = 'Bit Score')
            plt.tight_layout()
    
            #plt.savefig(path.split('/')[0] + 'RMSD_PC.png')
            plt.savefig(pdb1_name + '_RMSD_PC.png')
            #plt.clf()




            if int(total_target_count) == 1 and yyy == pdb1_name[:4]:
                print("Foldseek searched the single protein")

            elif int(total_target_count) > 1:

                for t in depth_sel:
                    print(t)
                    axes_depth.append(ax.scatter(x=df_phenix.loc[df_phenix['depth'].eq(t), 'RMSD PC 1'],
                                           y=df_phenix.loc[df_phenix['depth'].eq(t), 'RMSD PC 2'],
                                           s=65, marker="*", linewidth=0.5, linestyle="solid", edgecolors="white"))

                depth_plt = plt.legend(axes_depth, depth_sel, loc='upper right')
                plt.gca().add_artist(depth_plt)


                plt.legend(df_phenix['target'].unique(),loc='center left', bbox_to_anchor=(1.25, 0.5))
                plt.xticks(fontsize=15); plt.yticks(fontsize=15)

                #plt.colorbar(contour, label = 'Bit Score')
                #plt.tight_layout()

                #plt.savefig(path.split('/')[1] + 'RMSD_PC_depth.png')
                plt.savefig(pdb1_name + '_RMSD_PC_depth.png')
                plt.clf()

            elif len(target_count) == 1 and yyy == pdb1_name[:4]:
                print("Foldseek searched the single protein")


            else:
                print("Foldseek searched the single protein")













        #x_adjust = abs(min((df_phenix['RMSD PC 1'].min() * 0.25, df_phenix['RMSD PC 1'].max() * 0.25)))
        #y_adjust = abs(min((df_phenix['RMSD PC 2'].min() * 0.25, df_phenix['RMSD PC 2'].max() * 0.25)))
        #x_min = int(df_phenix['RMSD PC 1'].min() - x_adjust); x_max = int(df_phenix['RMSD PC 1'].max() + x_adjust)
        #y_min = int(df_phenix['RMSD PC 2'].min() - y_adjust); y_max = int(df_phenix['RMSD PC 2'].max() + y_adjust)

        #z = df_phenix['bits'].to_numpy()
        #x = df_phenix['RMSD PC 1'].to_numpy()
        #y = df_phenix['RMSD PC 2'].to_numpy()
        #z = z.astype('f'); x = x.astype('f'); y = y.astype('f')
        #x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 2000), np.linspace(y_min,y_max, 2000))
        #z_grid = np.zeros(x_grid.shape)
        ## vmax = np.max(z); vmin = np.min(z)
        ##vmax excludes bit scores from the full msa
        #vmax = df_phenix[df_phenix['Path'].map(lambda x:'Full' not in  x) == True]['bits'].map(lambda x: int(x)).max()
        #tree = cKDTree(list(zip(x,y)))
        #for (_x, _y, _z) in zip(x, y, z):
        #    _ = gauss(x_grid, y_grid, _x, _y, _z)  #generate gaussian
        #    _ = (_ / np.max(_))
        #    # _[_ < 0.5] = 0                         #makes the plot look more bubbly
        #    #scale by current value of z
        #    _ = _ * float(_z) if _z <= vmax else _ * float(vmax)
        #    dist, idx = tree.query((_x,_y), k=5)
        #    # reduce intensity of isolated configurations
        #    if sum(i > 2.0 for i in dist) > 1:
        #        #_ = _ * 0.5
        #        _ = _ * 0.01
        #    z_grid = np.maximum(z_grid, _)         #don't allow gaussians to build
        #contour = plt.contourf(x_grid, y_grid, z_grid, levels = 100, cmap='Greys')


        #cmap = cm._colormaps['rainbow_r']
        #norm = plt.Normalize(0, 100)
        #axes = []
        ##each 'type' needs its own handle for matplotlib to give unique legend elements
        #color_dict = defaultdict(int)
        #for idx, t in enumerate(df_phenix['target'].unique()):
        #    color_dict[t] = 100 * (idx / len(df_phenix['target'].unique()))
        #df_phenix['target_color'] = df_phenix['target'].map(lambda x: color_dict[x])


        #for idx, t in enumerate(df_phenix['target'].unique()):
        #    color_dict[t] = 100 * (idx / len(df_phenix['target'].unique()))
        #df_phenix['target_color'] = df_phenix['target'].map(lambda x: color_dict[x])

        #for t in df_phenix['target'].unique():
        #    axes.append(ax.scatter(x=df_phenix.loc[df_phenix['target'].eq(t), 'RMSD PC 1'], 
        #                           y= df_phenix.loc[df_phenix['target'].eq(t), 'RMSD PC 2'],
        #                           c=df_phenix.loc[df_phenix['target'].eq(t), 'target_color'].map(lambda x: cm.colors.rgb2hex(cmap(norm(x)), keep_alpha=True)),
        #                           s=45, linewidth=0, linestyle="None"))
        #plt.xlabel('RMSD PC 1'); plt.ylabel('RMSD PC 2')
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height* 0.80])
        #plt.legend(df_phenix['target'].unique(),loc='center left', bbox_to_anchor=(1.25, 0.5))


        #plt.colorbar(contour, label = 'Bit Score')
        #plt.tight_layout()

        ##plt.savefig(path.split('/')[0] + '_RMSD_PC.png')
        #plt.savefig(pdb1_name + '_RMSD_PC.png')
        #plt.clf()

