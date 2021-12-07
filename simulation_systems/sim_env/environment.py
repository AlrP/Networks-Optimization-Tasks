import datetime
import tqdm
import pandas as pd
import networkx as nx
import threading
import logging

import matplotlib.pyplot as plt
import visdom_utils as vu


class ClaimsAggregator:
    """
    Сборщик требований и статистики по ходу имитации
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: словарь с параметрами
            start_time: время начала
            simulation_time_passed: время, прошедшее с начала суток
            total_simulation_time_passed: общее прошедшее время симуляции
            days_passed: сколько дней симуляции прошло
            stop_simulation: нформация о работе симуляции
            claims_box: сборщик всех требований
            claims_processing: получено требований на обработку
            claims_done: обслужено требований
            claims_done_box: сборщик сделанных заказов
            claims_loss_box: сборщик потерянных заказов
            node_claims_loss: упущено требований в узле действия агента
            all_claims_loss: упущенно требований по всем узлам
            revenue: текущая выручка
            agent_capacity: текущие ресурсы агента
            agent_pos: текущая позиция агента
            agent_behaviour: словарь с временем отдыха агента и параметрами его среднего времени обслуживания
                             в заивисмости от частей суток
            agent_work: флаг, работает или нет агент
            agent_cur_mean_time: текущее среднее время соблуживания агента
            revenue_statistic: датасет с итоговой статистикой
            customers_intensity: словарь с фактическими потоками заказов клиентов по часам
            df_total_intensity_info: обработанные данные из customers_intensity
        """
        super(ClaimsAggregator, self).__init__()
        self.kwargs = kwargs
        self.start_time = None
        self.simulation_time_passed = 0
        self.total_simulation_time_passed = 0
        self.days_passed = 1
        self.stop_simulation = False
        self.claims_box = []
        self.claims_processing = 0
        self.claims_done = 0
        self.claims_done_box = []
        self.claims_loss_box = []
        self.node_claims_loss = 0
        self.all_claims_loss = 0
        self.revenue = 0
        self.agent_capacity = kwargs['init_agent_capacity']
        self.agent_pos = kwargs['init_agent_pos']
        self.agent_behaviour = kwargs['init_agent_behaviour']
        self.agent_work = False
        self.agent_cur_mean_time = None
        self.revenue_statistic = pd.DataFrame(columns=['total_time',
                                                       'model_time',
                                                       'model_day',
                                                       'agent_position',
                                                       'revenue',
                                                       'claims_processing',
                                                       'claims_done',
                                                       'node_claims_loss',
                                                       'all_claims_loss',
                                                       'resources'])
        self.customers_intensity = dict([(i + 1, dict([(i, 0) for i in range(24)]))
                                         for i in range(kwargs['customer_nodes'])])
        self.df_total_intensity_info = pd.DataFrame([])

        self.mutex = kwargs['mutex']
        self.scale_coef = kwargs['scale_coef']

    def update_sim_statistic(self):
        """
        Обновление симуляционной статистики
        """
        self.revenue_statistic = self.revenue_statistic.append(
            pd.DataFrame([{'total_time': self.total_simulation_time_passed,
                           'model_time': self.simulation_time_passed * self.scale_coef,
                           'model_day': self.days_passed,
                           'agent_position': self.agent_pos,
                           'revenue': float(self.revenue),
                           'claims_processing': self.claims_processing,
                           'claims_done': self.claims_done,
                           'node_claims_loss': self.node_claims_loss,
                           'all_claims_loss': self.all_claims_loss,
                           'resources': self.agent_capacity
                           }]))
        self.revenue_statistic.reset_index(inplace=True, drop=True)

    def update_claims_processing(self):
        """
        Обновить список заявок. Проверить, есть ли заявки, которых не дождался клиент и выкинуть их
        """
        self.mutex.acquire()
        # Добавление потерянных заявок
        # print('Lost:',self.claims_loss_box)
        self.claims_loss_box += [x for x in self.claims_box if
                                 (datetime.datetime.now() - x['time']).total_seconds() > x['patience']]
        # print([(datetime.datetime.now()-x['time']).total_seconds() for x in self.claims_box])
        # Очистка текущих
        self.claims_box = [x for x in self.claims_box if x not in self.claims_loss_box]
        # Текущее число заявок
        self.claims_processing = len([x['position'] == self.agent_pos for x in self.claims_box])
        # Обновлене упущенных заявок
        self.all_claims_loss = len([x for x in self.claims_loss_box])
        self.node_claims_loss = len([x for x in self.claims_loss_box if x['position'] == self.agent_pos])
        self.mutex.release()

    def update_add_intensity_info(self):
        """
        Дополнительное обновление интенсивности входящего потока за час по каждому дню
        """
        # Если наступает конец дня, добавить статистику в датасет и обнулить словарь
        for k, v in self.customers_intensity.items():
            df_tmp = pd.DataFrame([v])
            df_tmp['day'] = self.days_passed
            df_tmp['pos'] = k
            self.df_total_intensity_info = self.df_total_intensity_info.append(df_tmp)
            self.df_total_intensity_info.reset_index(inplace=True, drop=True)
            self.customers_intensity = dict([(i + 1, dict([(i, 0) for i in range(24)]))
                                             for i in range(self.kwargs['customer_nodes'])])

    def update_new_day(self):
        """
        Если наступил новый день, обнулить время
        """
        if self.simulation_time_passed > 23.9:
            self.update_add_intensity_info()
            self.simulation_time_passed = 0
            self.days_passed += 1
            logging.info(f'Day: {self.days_passed}'.center(40, '-'))

    def check_agent_work_hours(self):
        """
            Проверка поведения агента. В случае, если суточное время не совпадает с его рабочим,
        то действия агента блокируются, он уходит на отдых.
        """
        self.mutex.acquire()
        if (self.simulation_time_passed > self.agent_behaviour['rest_time_hours'][1]) & \
                (self.simulation_time_passed < self.agent_behaviour['rest_time_hours'][0]):
            self.agent_work = True
        else:
            self.agent_work = False
        self.mutex.release()

    def run_time(self, start_time, total_simulation_time):
        """
        Отсчет времени симуляции
        Args:
            start_time: время начала симуляции
            total_simulation_time: общее ограничение по времени симуляции
        """
        logging.info(f'Day: {self.days_passed}'.center(40, '-'))
        pbar = tqdm.tqdm_notebook(total=total_simulation_time,
                                  desc=f'Model time passed: {self.simulation_time_passed}',
                                  position=0, leave=True)
        t_0, t_1 = 0, 0
        while self.total_simulation_time_passed <= total_simulation_time:
            t_1 = datetime.datetime.now()

            self.update_claims_processing()

            if t_0 == 0:
                delta = float((t_1 - start_time).total_seconds())
            else:
                delta = float(((t_1 - t_0).total_seconds()))
                # Если текущее время от предыдущего отличается незначительно, пропустить обновление
                if (t_1 - t_0).total_seconds() < 1e-10:
                    pass
                else:
                    self.update_sim_statistic()

            self.simulation_time_passed += delta
            self.total_simulation_time_passed += delta

            pbar.update(delta)
            real_seconds = round(self.simulation_time_passed, 2)
            model_seconds = round(self.simulation_time_passed * self.scale_coef, 2)
            pbar.set_description(f"Seconds: [Real: {real_seconds} ][Model:{model_seconds}]")

            t_0 = t_1
            self.update_new_day()
            self.check_agent_work_hours()

        self.revenue_statistic = self.revenue_statistic.astype(dtype={
            'total_time': 'float64',
            'model_time': 'float64',
            'model_day': 'int64',
            'agent_position': 'int64',
            'revenue': 'float64',
            'claims_processing': 'int64',
            'claims_done': 'int64',
            'node_claims_loss': 'int64',
            'all_claims_loss': 'int64',
            'resources': 'int64'
        })
        self.stop_simulation = True
        print(f'[Trigger]: ClaimsAggregator.stop_simulation: {self.stop_simulation}')

    def start(self, start_time, total_simulation_time):
        """
        Используем threading для параллелизации процесса
        Args:
            start_time: время начала
            total_simulation_time: общее время симуляции
        """
        aggregator_thread = threading.Thread(target=self.run_time,
                                             args=[start_time, total_simulation_time], daemon=True)
        aggregator_thread.start()


class ModelEnvironment:
    """
    Общие параметры среды, в которой работает агент.
    """

    def __init__(self, mat, point_labels, scale_coef, agent_position=1,
                 print_logs_flag=True, visdom=True,
                 viz=False, connection='default'):
        """
        Args:
            print_logs_flag: флаг, печатать или нет события
            visdom: инстанс веб-сервера визуализации
            mat: структура, описывающая граф
            point_labels: название узлов графа
            scale_coef: коэффициент масштаба времени
            agent_position: текущая позиция агента
            env_graph: созданный граф
            start_time: время начала симуляции
            pos: для отрисовки графа, расположение узлов
        """
        super(ModelEnvironment, self).__init__()
        self.print_logs_flag = print_logs_flag
        self.visdom = visdom
        self.mat = mat
        self.point_labels = point_labels
        self.scale_coef = scale_coef
        self.agent_position = agent_position
        # ---------------------------------------
        self.env_graph = nx.MultiGraph()
        self.env_graph.add_edges_from(self.mat)
        self.start_time = datetime.datetime.now()
        self.pos = None
        # Интерактивные графики
        if self.visdom:
            self.visdom_instance = vu.VisdomLinePlotter(connection, viz=viz)
            self.visdom_instance.create_visdom_plots(self.point_labels)

    def plot_env_graph(self, figsize=(8, 6), node_clr='#1e78e6',
                       fnt_size=14, nd_size=900, width=3):
        """
        Прорисовка графа заданной рабочей среды
        Args:

        """
        fig = plt.figure(figsize=figsize)
        nx.draw_networkx(self.env_graph, self.pos, labels=self.point_labels,
                         with_labels=True, width=width, node_size=nd_size,
                         node_color=node_clr, font_size=fnt_size)
        plt.axis('off')
        plt.title('Точки перемещения мобильного пункта обслуживания', fontsize=16)
        plt.show()

    def run_simulation(self, **kwargs):
        """
        Запуск процесса симуляции
        """
        logging.info('Start Simulation'.center(80, r'='))
        # Ограничение по времени
        timelimit = kwargs['total_time']
        print(f'Real time limit: [{timelimit} sec]')
        print(f'Model time limit [{timelimit * self.scale_coef} sec][{timelimit * self.scale_coef / 3600} hour]')

        # Запуск инстанса ClaimsAggregator
        claims_aggr_instance = kwargs['claims_aggregator']
        claims_aggr_instance.start_time = self.start_time
        claims_aggr_instance.start(datetime.datetime.now(), timelimit)

        # Запуск инстансов PossibleCustomer
        for customer in kwargs['customer_types']:
            customer.start(claims_aggr_instance)

        # Запуск инстанса MobileAgent
        mobile_agent = kwargs['mobile_agent']
        mobile_agent.start(claims_aggr_instance)

        # Обновление интерактивных графиков процесса
        if self.visdom:
            self.visdom_instance.start(claims_aggr_instance, kwargs['customer_types'])
        # Логгирование
        if self.print_logs_flag:
            kwargs['logs'].print_logs(claims_aggr_instance)

        logging.shutdown()
