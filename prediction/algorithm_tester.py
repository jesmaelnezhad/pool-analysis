from utility import logger


class AlgorithmTester:
    def __init__(self, average_luck_windows, assessment_windows, lucks, assessments):
        self.average_luck_windows = average_luck_windows
        self.assessment_windows = assessment_windows
        self.lucks = lucks
        self.assessments = assessments
        self.algorithms = []

    def add_algorithm(self, algorithms):
        self.algorithms = self.algorithms + algorithms

    def test_algorithms(self, test_size=700, decision_aggregation_method="and",
                        no_decision_action="any-ignore", max_horizon=9 * 6):
        """

        :param test_size: number of data points to reserve for test
        :param decision_aggregation_method: and, anyX (at least X algorithms must return true)
        :param no_decision_action: any-ignore if any algorithm is no decision, the test will be ignored
        :return:
        """
        for alg in self.algorithms:
            alg.init()
        train_size = int(len(self.lucks) - test_size)
        horizon_total_counts = [0 for w in self.assessment_windows if w <= max_horizon]
        horizon_success_counts = [0 for w in self.assessment_windows if w <= max_horizon]
        horizon_positive_counts = [0 for w in self.assessment_windows if w <= max_horizon]
        horizon_real_positive_counts = [0 for w in self.assessment_windows if w <= max_horizon]
        horizon_positive_success_counts = [0 for w in self.assessment_windows if w <= max_horizon]
        horizon_algorithm_test_predictions = [[] for i in range(0, test_size)]
        for alg in self.algorithms:
            alg.fit(self.average_luck_windows,
                    self.assessment_windows, until=train_size, max_horizon=max_horizon)
            test_horizon_predictions = alg.predict(self.average_luck_windows,
                                                   self.assessment_windows, from_idx=train_size)

            for test_number in range(0, test_size):
                horizon_algorithm_test_predictions[test_number].append(test_horizon_predictions[test_number])
        for test_number in range(0, test_size):
            horizon_alg_predictions = horizon_algorithm_test_predictions[test_number]
            for i, w in enumerate(self.assessment_windows):
                if w > max_horizon:
                    continue
                horizon_decision = None
                if decision_aggregation_method == "and":
                    horizon_decision = True
                elif "any" == decision_aggregation_method[0:3]:
                    horizon_decision = False

                should_ignore_test = False
                accepted_count = 0
                for alg_idx, alg in enumerate(self.algorithms):
                    horizon_decision_tmp = alg.decide(self.lucks[train_size + test_number - 1],
                                                      self.average_luck_windows,
                                                      w,
                                                      horizon_alg_predictions[alg_idx],
                                                      self.assessment_windows)
                    logger("ALG-TESTER").debug("Horizon: {} / Algorithm: {} / Test number: {} / Decision: {}".format(
                        w,
                        alg_idx,
                        test_number,
                        horizon_decision_tmp
                    ))

                    if no_decision_action == "any-ignore" and horizon_decision_tmp is None:
                        should_ignore_test = True
                        break
                    if decision_aggregation_method == "and":
                        if horizon_decision_tmp is not True:
                            horizon_decision = False
                            break
                    elif "any" == decision_aggregation_method[0:3]:
                        if horizon_decision_tmp is True:
                            accepted_count += 1
                if should_ignore_test:
                    continue
                horizon_total_counts[i] += 1
                horizon_result = self.assessments[train_size + test_number - 1][i + 1] >= 1
                if horizon_result is True:
                    horizon_real_positive_counts[i] += 1

                if decision_aggregation_method == "and":
                    pass
                elif "any" == decision_aggregation_method[0:3]:
                    threshold = int(decision_aggregation_method[3:])
                    if accepted_count >= threshold:
                        horizon_decision = True
                logger("ALG-TESTER").debug("Horizon: {} / Test number: {} / D|{} vs. {}|R".format(
                    w,
                    test_number,
                    horizon_decision,
                    horizon_result
                ))
                if horizon_decision:
                    horizon_positive_counts[i] += 1
                if horizon_result == horizon_decision:
                    horizon_success_counts[i] += 1
                    if horizon_decision:
                        horizon_positive_success_counts[i] += 1
        assessment_windows_results = []
        for i, w in enumerate(self.assessment_windows):
            if w > max_horizon:
                continue
            total = horizon_total_counts[i] if horizon_total_counts[i] > 0 else 1
            assessment_windows_results.append((horizon_success_counts[i] * 100.0 / total,
                                               horizon_positive_counts[i] * 100.0 / total,
                                               horizon_real_positive_counts[i] * 100.0 / total,
                                               horizon_positive_success_counts[i] * 100.0 /
                                               (horizon_positive_counts[i] if horizon_positive_counts[
                                                                                  i] > 0 else 0.00001),
                                               horizon_positive_success_counts[i] * 100.0 / total,
                                               total))
        return assessment_windows_results
