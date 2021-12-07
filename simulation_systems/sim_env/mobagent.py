import time
import random
import numpy as np
import copy
import threading
import logging
import utils as ut


class MobileAgent:
    """
    Мобильный агент, перемещающийся между точками и исполняющий заказы
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: словарь с аргументами
            agent_position: позиция агента
            capacity: количество доступных ресурсов
            price: цена продажи заказа
            time_between_service: стартовое среднее время обслуживания заказа
            cur_time_between_service: текущеее среднее врем обслуживания
            env_graph: граф, представляющей собой торговое пространство
            position_strategy: конкретная стратегия перемещения по узлам
            stock_lag: сколько секунд длится загрузка новой партии ресурсов
            eps_capacity: при каком пороге ресурсов начать сворачивать работу и перемещаться к складу
            behaviour: словарь, определяющий рабочие часы и среднее время обслуживания в зависимости от частей суток
            work_flag: флаг, работает или нет агент
            strategy_flag: флаг типа стратегии работы агента
            plan_stage: стадия планирования перемещения, для delta_strategy
            node_loops: словарь-индикатор, сколько раз по каждому узлу прошелся агент, для delta_strategy
            estimates_array: матрица интенсивностей потока заказов в час (Часы x Узлы), для bayes_rule
            priors: словарь априорных распределений интенсивностей потока заказов по часам, для bayes_rule
            posterior: словарь апостериорных распределений интенсивностей потока заказов по часам, для bayes_rule
            bayes_transition: словарь переходов в узлы относительного каждого часа,
                              заданные через estimates_array, для bayes_rule

        """
        super(MobileAgent, self).__init__()
        self.kwargs = kwargs
        self.agent_position = copy.deepcopy(kwargs['start_position'])
        self.capacity = kwargs['capacity']
        self.price = kwargs['price']
        self.time_between_service = kwargs[
                                        'time_between_service_sec'] * kwargs['model_sec']
        self.cur_time_between_service = self.time_between_service
        self.env_graph = kwargs['env_graph']
        self.position_strategy = kwargs['position_strategy']
        self.stock_lag = kwargs['stock_lag']
        self.eps_capacity = kwargs['eps_capacity']
        self.behaviour = kwargs['behaviour']
        self.work_flag = False
        self.strategy_flag = kwargs['strategy_flag']
        self.plan_stage = 0
        self.node_loops = dict([(i + 1, 1)
                                for i in range(kwargs['customer_nodes'])])
        self.delta_revenue_tmp_collector = 0
        self.daily_revenue_collector = 0

        self.mutex = kwargs['mutex']
        self.scale_coef = kwargs['scale_coef']

        def init_prior(estimates_intens, kwargs):
            """
                Инициализация априорных знаний
            и использование их для первичного обновления оценок.
            Args:
                estimates_intens: матрица интенсивностей потока заказов в час (Часы x Узлы)
                kwargs: cсловарь ключевых аргументов, описывающих априорное распределение
            """
            # k - узел, v-словарь с alpha, beta для этого узла
            # +=, чтобы оставить случайность при инициализации
            for node, hour_ab in kwargs.items():
                for hour, v in hour_ab.items():
                    estimates_intens[hour, node - 1] += v['a'] / v['b']
            return estimates_intens

        self.estimates_array = np.random.rand(24, kwargs['customer_nodes'])
        self.priors = {i + 1:
                           {j: {'a': kwargs['bayes_prior']['a'],
                                'b': kwargs['bayes_prior']['b']} for j in range(24)}
                       for i in range(kwargs['customer_nodes'])}

        self.posterior = {i + 1:
                              {j: {'a': kwargs['bayes_prior']['a'],
                                   'b': kwargs['bayes_prior']['b']} for j in range(24)}
                          for i in range(kwargs['customer_nodes'])}

        self.estimates_array = init_prior(self.estimates_array, self.priors)
        self.bayes_transition = dict()
        print(f'Agent strategy: {self.strategy_flag}'.center(50, '='))

    def serve_claim(self, claims_aggregator):
        """
        Обслужить заявку из тех, что находятся в локации агента
        """
        tmp_claims_box = copy.deepcopy(claims_aggregator.claims_box)
        # Список требований только в узле агента
        filtered_claims = [
            x for x in tmp_claims_box if x['position'] == self.agent_position
        ]

        # Если требования есть, то обновляем claims_box, claims_done_box, claims_done
        if len(filtered_claims) > 0:
            # First-in-first-out (FIFO) выход заявок [0 - FIFO, -1 - LIFO]
            claim_to_done = filtered_claims[0]
            # Продолжительность обслуживания
            time.sleep(random.expovariate(1 / self.cur_time_between_service))
            # Удаление лишнего требования
            claims_aggregator.claims_box = [
                x for x in claims_aggregator.claims_box if x != claim_to_done
            ]
            # Добавление выполненного требования в список выполненных
            claims_aggregator.claims_done_box.append(claim_to_done)
            # Получение денег за выполнение заказа
            claims_aggregator.revenue += self.price
            # Для delta_strategy
            self.delta_revenue_tmp_collector += self.price
            self.daily_revenue_collector += self.price
            # Удаление единицы ресурса за исполнение заказа
            self.capacity -= 1
            claims_aggregator.agent_capacity -= 1
            # Отметка о выполнении заказа в агрегаторе
            claims_aggregator.claims_done += 1
            # print('Обработано: ', claims_aggregator.claims_done)

    def do_nothing(self, time_length):
        """
        Ничего не делать в случае перемещения в новый узел или стоянии на складе
        Args:
            time_length: длина времени, в течении которого агент ничего не делает
        """
        do_nothing_time = time_length * self.kwargs['model_sec']
        logging.info(f'[MobileAgent] Do nothing: {time_length} model sec')
        time.sleep(random.expovariate(1 / (do_nothing_time)))

    def get_moving_mean_time(self, start_point, end_point):
        """
        Вернуть среднее время перехода из одной точки в другую
        Args:
            start_point: начальный узел
            end_point: конечный узел
        """
        graph_address = (start_point, end_point, 0)
        return self.env_graph.edges[graph_address]['mean_time']

    def check_capacity(self, claims_aggregator):
        """
        Проверить наличие возможности выполнения заказов
        Если нет, то change_position к узлу Stock
        Args:
            claims_aggregator: сборщик статистики имитационной модели
        """
        claims_aggregator.agent_capacity = self.capacity
        if self.capacity <= self.eps_capacity:
            self.change_position(4, claims_aggregator)
        else:
            pass

    def update_daily_strategy(self, claims_aggregator):
        """
        Ежедневное обновление стратегии передвижения, если таковая имеется
        Args:
            claims_aggregator: сборщик статистики имитационной модели
        """
        if self.strategy_flag == 'position_strategy':
            tp = claims_aggregator.simulation_time_passed
            # Обнуление стадии планирования
            if claims_aggregator.days_passed > 1:
                self.plan_stage = 0
                logging.info(
                    f'[MobileAgent][{round(tp * self.scale_coef, 3)}]Обновление плана работы агента'
                )
        elif self.strategy_flag == 'delta_strategy':
            # Обновим циклы прогона по узлам. не надо
            # self.node_loops = dict([(i+1, 1) for i in range(self.kwargs['customer_nodes'])])
            pass
        elif self.strategy_flag == 'bayes_rule':
            if claims_aggregator.days_passed > 1:
                # Обновим матрицу оценок интенсивностей, если прошел день после
                # инициализации первых априорных оценок
                self.update_posterior(claims_aggregator)
                self.update_estimates_array()
            # Просчет стратегии перемещения на день
            self.assign_nodes_to_hours()

    def plan_strategy(self, claims_aggregator):
        """
        Действия по запланирвоанной стратегии, если таковая имеется
        Args:
            claims_aggregator: сборщик статистики имитационной модели
        """
        # Если еще есть передвижения согласно стратегии
        if self.plan_stage + 1 <= len(self.position_strategy):
            # Если время исполнения стратегии наступило
            if claims_aggregator.simulation_time_passed * self.scale_coef >= \
                    self.position_strategy[self.plan_stage]['timeline']:
                # Выбрать новое местоположение, выбросив текущую стратегию из плана
                cur_strat = self.position_strategy[self.plan_stage]
                new_position = cur_strat['move_node']
                self.plan_stage += 1
            else:
                new_position = self.agent_position
        else:
            new_position = self.agent_position
        return new_position

    def delta_strategy(self, claims_aggregator):
        """
        Дельта стратегия достижения заданного уровня выручки по узлу
        Args:
            claims_aggregator: сборщик статистики имитационной модели
        """
        # Все узлы
        nodes = [i + 1 for i in range(self.kwargs['customer_nodes'])]
        # Уберем текущий
        nodes.remove(self.agent_position)
        # Размешаем оставшиеся
        random.shuffle(nodes)
        # Выручка
        if claims_aggregator.revenue_statistic.shape[0] > 0:
            self.mutex.acquire()
            cur_revenue = claims_aggregator.revenue_statistic[ \
                (claims_aggregator.revenue_statistic['model_day'] == claims_aggregator.days_passed) & \
                (claims_aggregator.revenue_statistic['agent_position'] == self.agent_position)]['revenue'].max()
            self.mutex.release()

            # Если текущая выручка по узлу за день>=Дельты*Число циклов по узлу
            if self.delta_revenue_tmp_collector > self.kwargs['delta_revenue'] + 0.5e1:
                info = round(
                    claims_aggregator.simulation_time_passed * self.scale_coef, 3)
                delta = self.kwargs['delta_revenue']
                logging.info(
                    f'[MobileAgent][{info}] Получена дельта выручки {delta}. Итого: {self.agent_position}: {cur_revenue}'
                )  # По узлу {self.agent_position}: {cur_revenue}
                # Обновить словарь пройденных узлов
                self.node_loops[self.agent_position] += 1
                logging.info(
                    f'[MobileAgent] Пройдено по узлам:{self.node_loops}')
                # Обнулить счетчик дельты-выручки:
                self.delta_revenue_tmp_collector = 0
                # Вернуть новое местоположение
                return nodes[0]
            else:
                # Вернуть текущее местоположение
                return self.agent_position
        else:
            # Вернуть текущее местоположение
            return self.agent_position

    def get_best_node(self, hour):
        """
        Лучший узел заданного часа по апостериорной плотностия вероятности.
        Args:
            hour: текущий час
        """
        return self.estimates_array[hour, :].argmax() + 1

    def assign_nodes_to_hours(self):
        """
            Определение часов на перемещение по байесовской оценке determined_work_hours
        и случайное переещение left_work_hours. Определяем план перемещения по узлам
        относительно каждого часа.
        """

        determined_work_hours_frac = self.kwargs['determined_work_hours_frac']

        daily_hours = [
            i
            for i in range(self.kwargs['behaviour']['rest_time_hours'][1],
                           self.kwargs['behaviour']['rest_time_hours'][0] + 1)
        ]
        random.shuffle(daily_hours)
        working_index = int(len(daily_hours) * determined_work_hours_frac)

        determined_work_hours = daily_hours[:working_index]
        left_work_hours = daily_hours[working_index:]

        for h in range(24):
            if h in determined_work_hours:
                # Испольузется оценка апостериорной плотности вероятности
                self.bayes_transition[h] = self.get_best_node(h)
            elif h in left_work_hours:
                # Случайный поведенческо-исследовательский аспект
                self.bayes_transition[h] = np.random.randint(
                    low=1, high=self.kwargs['customer_nodes'])
            else:
                # raise AssertionError(f'Hour {h} not in determined_work_hours/left_work_hours')
                # В эти часы агент просто не работает. Устанавливаем 1-й узел по умолчанию
                self.bayes_transition[h] = 1

    def update_posterior(self, claims_aggregator):
        """
        Обновить словарь параметров для апостериорной плотности вероятности в начале нового дня.
        Используются данные по self.bayes_transition, claims_aggregator.df_total_intensity_info
        Args:
            claims_aggregator: сборщик статистики для имитационной модели
        """
        cur_day = claims_aggregator.days_passed
        # Для часа, узла
        for hour, node in self.bayes_transition.items():
            # Берем статистику потока заявок за час предыдущего дня
            claims_in_hour = claims_aggregator.df_total_intensity_info[
                (claims_aggregator.df_total_intensity_info['pos'] == node) &
                (claims_aggregator.df_total_intensity_info['day'] == cur_day -
                 1)][hour].values[0]
            # Обновление alpha: new_alpha = alpha+sum(xi)
            self.posterior[node][hour]['a'] += sum([claims_in_hour])
            # Обновление beta: new_beta = beta + n
            self.posterior[node][hour]['b'] += len([claims_in_hour])

    def update_estimates_array(self):
        """
            Обновить матрицу оценок математических ожиданий
        апостериорных плотностей верояностей по всем часам всех узлов.
        Используется self.posterior и self.estimates_array
        """
        logging.info(f'[MobileAgent] Обновление знаний агента о среде.')
        for node, hour_ab in self.posterior.items():
            for hour, v in hour_ab.items():
                self.estimates_array[hour, node - 1] = v['a'] / v['b']

    def bayes_rule(self, claims_aggregator):
        """
        Вернуть лучший узел для работы из self.bayes_transition относительно текущего времени
        0: лучший узел для времени [0, 1)
        1: лучший узел для времени [1, 2)
        ...
        23: лучший узел для времени [23, 23.(9))
        Args:
            claims_aggregator: сборщик статистики для имитационной модели
        """
        cur_hour = round(claims_aggregator.simulation_time_passed)
        return self.bayes_transition[0 if cur_hour >= 24 else cur_hour]

    def agent_strategy(self, claims_aggregator):
        """
        Обертка над функциями plan_strategy, delta_strategy, bayes_rule
        Args:
            claims_aggregator: сборщик статистики для имитационной модели
        """
        if self.strategy_flag == 'position_strategy':
            pos = self.plan_strategy(claims_aggregator)
        elif self.strategy_flag == 'delta_strategy':
            pos = self.delta_strategy(claims_aggregator)
        elif self.strategy_flag == 'bayes_rule':
            pos = self.bayes_rule(claims_aggregator)
            pass
        else:
            raise AssertionError(
                f'Strategy {self.strategy_flag} is not defined')
        return pos

    def change_position(self, new_position, claims_aggregator):
        """
        Поставить невозможность обслуживания, затем изменить локацию
        Args:
            new_position: узел перехода агента
            claims_aggregator: сборщик статистики для имитационной модели
        """
        if new_position == self.agent_position:
            pass
        elif new_position == 4:
            # Передвижение на склад
            tp = claims_aggregator.simulation_time_passed
            logging.info(
                f'[MobileAgent][{round(tp * self.scale_coef, 3)}]Внимание. Ресурсов осталось: {self.capacity}'
            )
            logging.info(
                f'[MobileAgent][{round(tp * self.scale_coef, 3)}] Передвижение {self.agent_position}->Склад.'
            )
            self.do_nothing(self.get_moving_mean_time(self.agent_position, 4))
            # Загрузка запасов со склада
            logging.info(
                f'[MobileAgent][{round(tp * self.scale_coef, 3)}]Пополнение ресурсов.')
            self.do_nothing(self.stock_lag)

            self.capacity += self.kwargs['stock_update']
            self.check_capacity(claims_aggregator)

            logging.info(
                f'[MobileAgent][{round(tp * self.scale_coef, 3)}]Ресурсы пополнены. Наличие:{self.capacity}'
            )
            logging.info(
                f'[MobileAgent][{round(tp * self.scale_coef, 3)}] Передвижение Склад->{self.agent_position}.'
            )
        else:
            tp = claims_aggregator.simulation_time_passed
            logging.info(
                f'[MobileAgent][{round(tp * self.scale_coef, 3)}] Передвижение {self.agent_position}->{new_position}.'
            )
            self.do_nothing(
                self.get_moving_mean_time(self.agent_position, new_position))
            claims_aggregator.agent_pos = new_position
            self.agent_position = new_position

    def work_constraint(self, claims_aggregator):
        """
        Условие проверки работы агента
        Args:
            claims_aggregator: сборщик статистики для имитационной модели
        """
        tp = claims_aggregator.simulation_time_passed
        # Если агент не должен работать
        if not claims_aggregator.agent_work:
            # Если метка стоит, что он не работает
            if not self.work_flag:
                # Вернуть False - дальнейшая работа продолжается
                return False
            else:
                # Если метка стоит, что он работает
                logging.info(
                    f'[MobileAgent][{round(tp * self.scale_coef, 3)}]Агент уходит отдыхать.'
                )
                logging.info(
                    f'[MobileAgent][Выручка агента: {self.daily_revenue_collector}].'
                )
                self.daily_revenue_collector = 0
                # Ставим метку, что он не работает
                self.work_flag = False
                return False
        # Если агент должен работать
        else:
            if claims_aggregator.agent_work:
                # Если метка стоит, что он работает
                if self.work_flag:
                    # Вернуть True - дальнейшая работа продолжается
                    return True
                else:
                    # Если метка стоит, что он не работает
                    logging.info(
                        f'[MobileAgent][{round(tp * self.scale_coef, 3)}]Агент начинает работу.'
                    )
                    self.work_flag = True
                    # Обнуление порядка стратегий
                    self.update_daily_strategy(claims_aggregator)
                    return True

    def update_working_speed(self, claims_aggregator):
        """
        Обновить среднее время обслуживания в зависимости от внутрисуточного времени
        Args:
            claims_aggregator: сборщик статистики для имитационной модели
        """
        mtime = ut.normalized_sigmoid_fkt(
            self.behaviour['service_time']['a'],
            self.behaviour['service_time']['b'],
            claims_aggregator.simulation_time_passed / 24) * self.time_between_service
        self.cur_time_between_service = mtime + np.random.normal(0, 0.05 * mtime)
        claims_aggregator.agent_cur_mean_time = self.cur_time_between_service * self.scale_coef

    def run(self, claims_aggregator):
        """
        Запуск логики работы агента
        Args:
            claims_aggregator: сборщик статистики для имитационной модели
        """
        claims_aggregator.agent_capacity = self.capacity

        while not claims_aggregator.stop_simulation:
            # Проверка ограничения времени работы агента
            if self.work_constraint(claims_aggregator):
                # Обновление среднего времени обслуживания внутри суток
                self.update_working_speed(claims_aggregator)
                # Проверка ресурсов
                self.check_capacity(claims_aggregator)
                # Обработка заявки и запись в агрегатор
                self.serve_claim(claims_aggregator)
                # Передвижение согласно выбранной стратегии
                # TODO Обертка: 1.Работа по плану, 2. Работа по дельте выручки, 3.Работа по Байесу
                new_position = self.agent_strategy(claims_aggregator)
                # О стратегии перемещении агента
                self.change_position(new_position, claims_aggregator)

    def start(self, kwargs):
        """
        Используем threading для параллелизации процесса
        Args:
            kwargs: словарь с параметрами
        """
        magent_thread = threading.Thread(target=self.run,
                                         args=[kwargs],
                                         daemon=True)
        magent_thread.start()
