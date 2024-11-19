import matplotlib
import matplotlib.pyplot as plt


def define_plot_settings(font_size = 20):
        font = {'family' : 'serif', #'Times New Roman', #'sans-serif',
                # 'serif'  : 'Times New Roman',# 'sans-serif': ['Helvetica'],
                # 'serif'  : 'DejaVu Serif',# 'sans-serif': ['Helvetica'],
                'size'   : font_size}
        matplotlib.rc('font', **font)
        matplotlib.rc('mathtext', **{'fontset': 'dejavuserif'})
        SMALL_SIZE =  font_size # - 2
        MEDIUM_SIZE = font_size
        BIGGER_SIZE = font_size #+ 2
        plt.rc('font', size=SMALL_SIZE)          # Controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # Font size of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # Font size of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # Font size of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # Font size of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # Legend font size
        plt.rc('figure', titlesize=BIGGER_SIZE)  # Fontsize of the figure title
        plt.rcParams.update({
                "text.usetex": False,
                'text.latex.preamble':  r'\usepackage{amsmath}'
                                        r'\usepackage{amssymb}'                                                
                                        r'\usepackage{tgheros}'    
                                        # r'\usepackage{sansmath}'
                                        # r'\sansmath'      
                                #        r'\usepackage{commath}',
        }) 