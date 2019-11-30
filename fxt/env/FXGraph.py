import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class FXGraph:

    def __init__(self, df, title=None):
        self.df = df
        self.net_worths = np.zeros(len(df))

        fig = plt.figure()
        fig.suptitle(title)
        
        self.net_worth_ax = plt.subplot2grid((5, 1), (0, 0),
                rowspan=2, colspan=1)

        self.price_ax = plt.subplot2grid((5, 1), (2, 0),
                rowspan=5, colspan=1)

        self.volume_ax = self.price_ax.twinx()

        plt.subplots_adjust(left=0.11, bottom=0.15, right=0.90,
                top=0.90, wspace=0.2, hspace=0)

        plt.show(block=False)