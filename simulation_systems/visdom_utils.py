
import threading
import time
import numpy as np
import datetime

from visdom import Visdom
from collections import Counter

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='simulation_project', viz=False, connection='default'):
        super(VisdomLinePlotter, self).__init__()
        self.connection = connection
        self.env = env_name

        if viz==False:
            print(f'{datetime.datetime.now()} Visdom connection: {self.connection}')
            if self.connection=='default':
                self.viz = Visdom(env=self.env)
            else:
                self.viz = Visdom(server=self.connection,
                                env=self.env)
        else:
            self.viz = viz

        self.plots = {}

        assert self.viz.check_connection()
        
    def create_plots(self, n_plots):
        # Задаем все графики
        # График заявок, график исполненных заявок, график выручки, график пропущенных заявок, график ресурсов
        print('Visdom plots:')

        for plot_name, xlabel, ylabel, title, legend in zip(
            ['claims_plot',
             'done_claims_plot', 
             'missed_claims_plot',
             'agent_resources',
             'agent_service_time',
             'customer_models',
             'revenue_plot'
             ],
             ['Model Time' for x in range(7)],
             ['Claims', 'Claims','Claims', 'Resources', 'Service Mean Time','Spacings Mean Time', 'Money'],
             ['Claims Simulation', 'Done Claims', 'Missed Claims', 'Resources', 'Agent Model', 'Customer Models','Revenue'], 
             [[f'Node {i+1}' for i in range(n_plots)],
              [f'Node {i+1}' for i in range(n_plots)],
              [f'Node {i+1}' for i in range(n_plots)],
              ['Resources'],
              ['Agent Model'],
              [f'Node {i+1}' for i in range(n_plots)],
              [f'Node {i+1}' for i in range(n_plots)]]):
            print(f'{datetime.datetime.now()} {plot_name}')

            self.plots[plot_name] =self.viz.line(X=np.array([0 for x in range(n_plots)]),
                                                     Y=np.array([0 for x in range(n_plots)]),
                                                     opts=dict(
                                                        xlabel=xlabel,
                                                        ylabel=ylabel,
                                                        title=title,
                                                        legend=legend)
                                                     )
            # Time lag to let all the plots be posted on visdom web-server
            time.sleep(1.5)
 
        print(f'{datetime.datetime.now()} total_revenue')
        self.plots['total_revenue'] =self.viz.line(X=np.array([0]),
                                        Y=np.array([0]),
                                        opts=dict(
                                        xlabel='Model Time',
                                        ylabel='Total revenue',
                                        title='Total revenue',
                                        legend=['Total revenue'])
                                        )


    def update_plot(self, x, y, legend_name, plot_name):
        self.viz.line(X=x, Y=y, win=self.plots[plot_name], name=legend_name, update='append')

    def create_visdom_plots(self, point_labels):
        # Создание ряда графиков
        self.create_plots(len(point_labels.keys())-1)
    
    def update_visdom_plots(self, claims_aggregator, customer_list, price=1):
        # Обновление графиков
        # x, y, legend_name, plot_name
        while (claims_aggregator.stop_simulation == False):
        	# time.sleep(0.5)
            # claims_plot
            t = claims_aggregator.total_simulation_time_passed
            claims_dict = Counter(x['position'] for x in claims_aggregator.claims_box)
            claims_done_dict = Counter(x['position'] for x in claims_aggregator.claims_done_box)
            claims_loss_dict = Counter(x['position'] for x in claims_aggregator.claims_loss_box)

            customer_types_dict= dict([(x.position, x.cur_time_between_claims) for x in customer_list])

            for k, v in claims_dict.items():
                self.update_plot([t], [v], f'Node {k}', 'claims_plot')

            for k, v in claims_done_dict.items():
                self.update_plot([t], [v], f'Node {k}', 'done_claims_plot')

            for k, v in claims_loss_dict.items():
                self.update_plot([t], [v], f'Node {k}', 'missed_claims_plot')

            for k, v in claims_done_dict.items():
                self.update_plot([t], [v*price], f'Node {k}', 'revenue_plot')

            for k, v in customer_types_dict.items():
                self.update_plot([t], [v], f'Node {k}', 'customer_models')

            self.update_plot([t], [claims_aggregator.agent_capacity], 'Resources', 'agent_resources')
            self.update_plot([t], [claims_aggregator.agent_cur_mean_time], 'Agent Model', 'agent_service_time')
            self.update_plot([t], [len(claims_aggregator.claims_done_box)], 'Total revenue', 'total_revenue')


    def start(self, claims_aggregator, customer_list):
        """
        Используем threading для параллелизации процесса
        """
        visdom_thread = threading.Thread(target=self.update_visdom_plots, 
                                         args=[claims_aggregator, customer_list])
        visdom_thread.start()