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
        horizon_positive_success_counts = [0 for w in self.assessment_windows if w <= max_horizon]
        horizon_algorithm_test_predictions = [[] for i in range(0, test_size)]
        for alg in self.algorithms:
            alg.fit(self.average_luck_windows,
                    self.assessment_windows, until=train_size, max_horizon=max_horizon)
            test_horizon_predictions = alg.predict(self.average_luck_windows,
                                                   self.assessment_windows, from_idx=train_size)

            for test_number in range(0, test_size):
                horizon_algorithm_test_predictions[test_number].append(test_horizon_predictions[test_number])
                print(str(len(test_horizon_predictions[test_number])) + " ####")
        # print(str(len(test_horizon_predictions)))
        for test_number in range(0, test_size):
            print(str(test_number) + " / " + str(test_size) )
            horizon_alg_predictions = horizon_algorithm_test_predictions[test_number]
            print(str(horizon_alg_predictions))
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
                    # print(str(horizon_prediction))
                    horizon_decision_tmp = alg.decide(self.lucks[train_size + test_number - 1],
                                                      self.average_luck_windows,
                                                      w,
                                                      horizon_alg_predictions[alg_idx],
                                                      self.assessment_windows)
                    if no_decision_action == "any-ignore" and horizon_decision_tmp is None:
                        should_ignore_test = True
                        break
                    if decision_aggregation_method == "and":
                        if not horizon_decision_tmp:
                            horizon_decision = False
                            break
                    elif "any" == decision_aggregation_method[0:3]:
                        if horizon_decision_tmp:
                            accepted_count += 1
                if should_ignore_test:
                    continue
                horizon_total_counts[i] += 1
                horizon_result = self.assessments[train_size + test_number - 1][i + 1] >= 1
                # print(str(self.assessments[train_size + test_number - 1][i + 1]))
                if decision_aggregation_method == "and":
                    pass
                elif "any" == decision_aggregation_method[0:3]:
                    threshold = int(decision_aggregation_method[3:])
                    if accepted_count >= threshold:
                        horizon_decision = True

                if horizon_decision:
                    horizon_positive_counts[i] += 1
                if horizon_result == horizon_decision:
                    horizon_success_counts[i] += 1
                    if horizon_decision:
                        horizon_positive_success_counts[i] += 1
        ## print results
        # logger("RESULTS").info("            SUCCESS\tPOSITIVE\tPOSITIVE SUCCESS\tPOSITIVE SUCCESS TOTAL\tTotal")
        assessment_windows_results = []
        for i, w in enumerate(self.assessment_windows):
            if w > max_horizon:
                continue
            # logger("RESULTS").info(
            #     "Horizon {} : {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}".format(w, horizon_success_counts[i] * 100.0 / horizon_total_counts[i],
            #                                                          horizon_positive_counts[i] * 100.0 / horizon_total_counts[i],
            #                                                          horizon_positive_success_counts[i] * 100.0 /
            #                                                          (horizon_positive_counts[i] if
            #                                                           horizon_positive_counts[i] > 0 else 0.00001),
            #                                                          horizon_positive_success_counts[
            #                                                              i] * 100.0 / horizon_total_counts[i],
            #                                                              horizon_total_counts[i]))
            total = horizon_total_counts[i] if horizon_total_counts[i] > 0 else 1
            assessment_windows_results.append((horizon_success_counts[i] * 100.0 / total,
                                               horizon_positive_counts[i] * 100.0 / total,
                                               horizon_positive_success_counts[i] * 100.0 /
                                               (horizon_positive_counts[i] if horizon_positive_counts[
                                                                                  i] > 0 else 0.00001),
                                               horizon_positive_success_counts[i] * 100.0 / total,
                                               total))
        return assessment_windows_results
