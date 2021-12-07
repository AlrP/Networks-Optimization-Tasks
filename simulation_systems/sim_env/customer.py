import datetime
import time
import random
import scipy.stats as st
import numpy as np
import threading
import logging


class PossibleCustomer:
    """
    Модель потенциального покупателя.
    """

    def __init__(self, **kwargs):
        """
        Args:
        kwargs - словрь с аргументами
        name: имя условного покупателя
        position: позиция покупателя в графе
        patience: столько секунд готов ждать обслуживания в очереди
        time_between_claims: среднее время в секундах между
                             приходами запросов от потенциального покупателя
        history_id_claim: id запроса заказа
        """
        super(PossibleCustomer, self).__init__()
        self.kwargs = kwargs
        self.name = kwargs['name']
        self.position = kwargs['position']
        self.patience = kwargs['patience_sec']
        self.time_between_claims = kwargs['between_claims_sec'] * kwargs[
            'model_sec']
        self.cur_time_between_claims = self.time_between_claims
        self.history_id_claim = 0

        self.mutex = kwargs['mutex']
        self.scale_coef = kwargs['scale_coef']

        def get_timedetermine_distr(a, b):
            """
            Получение заданного распределения
            Args:
                a: alpha
                b: beta
            """
            x = np.linspace(st.beta.ppf(0.01, a, b), st.beta.ppf(0.99, a, b), 500)
            beta_rv = st.beta(a, b)
            max_ = max(beta_rv.pdf(x))
            return beta_rv, max_

        if kwargs['spacings_law']:
            self.spacings_law_param = kwargs['spacings_law']
            self.spacings_law, self.law_max = \
                get_timedetermine_distr(kwargs['spacings_law']['a'],
                                        kwargs['spacings_law']['b'])
        else:
            self.spacings_law_param = f'const {self.time_between_claims}'

    def get_cur_mean_time(self, time_passed_seconds):
        """
        Текущее среднее время промежутка между заказами
        Args:
            time_passed_seconds: прошедшее время за день
        """
        return (self.spacings_law.pdf(time_passed_seconds / (24 * 3600)) /
                self.law_max) * self.time_between_claims

    def update_mean_time(self, claims_aggregator):
        """
        Обновление среднего времени между запросами заказов
        Args:
            claims_aggregator: сборщик статистики имитационной модели
        """
        if self.spacings_law:
            self.mutex.acquire()
            mtime = self.get_cur_mean_time(
                claims_aggregator.simulation_time_passed *
                self.scale_coef) * self.scale_coef
            self.mutex.release()
            self.cur_time_between_claims = mtime + np.random.normal(0, 0.05 * mtime)
        else:
            self.cur_time_between_claims = self.time_between_claims

    def send_claim(self, claims_aggregator):
        """
        Отправка заявки на заказ
        Args:
            claims_aggregator: сборщик статистики имитационной модели
        """
        time.sleep(
            random.expovariate(
                1 / (self.cur_time_between_claims * self.kwargs['model_sec'])))
        self.history_id_claim += 1
        tmp_time_hour = round(claims_aggregator.simulation_time_passed)
        claims_aggregator.customers_intensity[
            self.position][0 if tmp_time_hour >= 24 else tmp_time_hour] += 1
        return {
            'id': self.history_id_claim,
            'time': datetime.datetime.now(),
            'patience': self.patience * self.kwargs['model_sec'],
            'position': self.position
        }

    def run(self, claims_aggregator):
        """
        Запуск логики работы модели покупателя
        Args:
            claims_aggregator: сборщик статистики имитационной модели
        """
        logging.info(
            f'[PossibleCustomer: {self.name}] Spacings law: {self.spacings_law_param}'
        )

        while not claims_aggregator.stop_simulation:
            # Обновить среднее время между заявками
            self.update_mean_time(claims_aggregator)
            # Добавление требования в список-атрибут в классе сборщика требований
            claims_aggregator.claims_box.append(
                self.send_claim(claims_aggregator))

    def start(self, claims_aggregator):
        """
        Используем threading для параллелизации процесса
        Args:
            claims_aggregator: сборщик статистики имитационной модели
        """
        customer_thread = threading.Thread(target=self.run,
                                           args=[claims_aggregator],
                                           daemon=True)
        customer_thread.start()
