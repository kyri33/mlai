import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_finance import candlestick_ochl as candlestick

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'

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

    def render(self, current_step, net_worth, trades,
            window_size = 120):
        
        self.net_worths[current_step] = net_worth
        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        # TODO Dates

        self._render_net_worth(current_step,
                step_range)
        self._render_price(current_step, step_range)
        self._render_trades(current_step, trades, step_range)

        plt.pause(0.001)
    
    def _render_net_worth(self, current_step,
            step_range):
        
        self.net_worth_ax.clear()
        
        self.net_worth_ax.plot(step_range, self.net_worths[step_range],
                '-', label='Net Worth')

        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2,
                prop={'size': 4})
        legend.get_frame().set_alpha(0.4)

        last_net_worth = self.net_worths[current_step]

        self.net_worth_ax.annotate('{0:.2f}'.format(last_net_worth),
                (current_step, last_net_worth), 
                xytext=(current_step, last_net_worth),
                bbox = dict(boxstyle='round', fc='w', ec='k', lw=1),
                fontsize='small')
        
        self.net_worth_ax.set_ylim(min(self.net_worths[np.nonzero(self.net_worths)] / 1.01),
                max(self.net_worths) * 1.01)

    def _render_price(self, current_step, step_range):
        self.price_ax.clear()

        candlesticks = zip(step_range, self.df['Open'].values[step_range],
                self.df['Close'].values[step_range],
                self.df['High'].values[step_range],
                self.df['Low'].values[step_range])
        candlestick(self.price_ax, candlesticks, width=1,
                colorup=UP_COLOR, colordown=DOWN_COLOR)
        
        last_close = self.df['Close'].values[current_step]
        last_high = self.df['High'].values[current_step]

        self.price_ax.annotate('{0:.2f}'.format(last_close),
                (current_step, last_close),
                xytext=(current_step, last_high),
                bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                color='black', fontsize='small')
    
    def _render_trades(self, current_step, trades, step_range):
        for trade in trades:
            if trade['step'] in step_range:
                high = self.df['High'].values[trade['step']]
                low = self.df['Low'].values[trade['step']]

                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_COLOR
                else:
                    high_low = high
                    color = DOWN_COLOR
                
                total = '{0:.3f}'.format(trade['total'])
                self.price_ax.annotate(f'${total}', 
                        (trade['step'], high_low),
                        color = color,
                        fontsize=8)
                        #arrowprops=(dict(color=color)))