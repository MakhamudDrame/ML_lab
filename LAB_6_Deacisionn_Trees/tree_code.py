import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:`R` — множество всех объектов,
    * :math:`R_l` и :math:`R_r` — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:`H(R)`:
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:`p_1` и :math:`p_0` — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в `thresholds`.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """

    # Сортируем признаки и таргеты
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    # Получаем уникальные значения признака
    unique_features = np.unique(sorted_features)

    # Генерируем кандидаты порогов как средние между соседними уникальными значениями
    if len(unique_features) <= 1:
        return np.array([]), np.array([]), None, None
    thresholds_candidates = (unique_features[:-1] + unique_features[1:]) / 2

    thresholds_list = []
    ginis_list = []
    best_gini = -np.inf
    best_threshold = None

    total_size = len(target_vector)

    for threshold in thresholds_candidates:
        # Определяем размер левой части
        left_size = np.searchsorted(sorted_features, threshold, side='left')
        right_size = total_size - left_size

        # Пропускаем пороги, которые приводят к пустым поддеревьям
        if left_size == 0 or right_size == 0:
            continue

        # Добавляем порог в список
        thresholds_list.append(threshold)

        # Подсчёт сумм классов 1 в левой и правой частях
        sum_left = np.sum(sorted_targets[:left_size])
        sum_right = np.sum(sorted_targets[left_size:])

        # Вычисление энтропии для левого поддерева
        if left_size == 0:
            H_left = 0
        else:
            p1_left = sum_left / left_size
            p0_left = 1 - p1_left
            H_left = 1 - p1_left**2 - p0_left**2

        # Вычисление энтропии для правого поддерева
        if right_size == 0:
            H_right = 0
        else:
            p1_right = sum_right / right_size
            p0_right = 1 - p1_right
            H_right = 1 - p1_right**2 - p0_right**2

        # Вычисление критерия Джини
        gini = - (left_size / total_size) * H_left - (right_size / total_size) * H_right
        ginis_list.append(gini)

        # Обновляем лучший порог
        if gini > best_gini or (gini == best_gini and threshold < best_threshold):
            best_gini = gini
            best_threshold = threshold

    # Если нет подходящих порогов
    if not thresholds_list:
        return np.array([]), np.array([]), None, None

    # Преобразуем в массивы numpy и сортируем
    thresholds_array = np.array(thresholds_list)
    ginis_array = np.array(ginis_list)

    sorted_indices = np.argsort(thresholds_array)
    thresholds_sorted = thresholds_array[sorted_indices]
    ginis_sorted = ginis_array[sorted_indices]

    return thresholds_sorted, ginis_sorted, best_threshold, best_gini


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        """
        Обучение узла дерева решений.

        Если все элементы в подвыборке принадлежат одному классу, узел становится терминальным.

        Parameters
        ----------
        sub_X : np.ndarray
            Подвыборка признаков.
        sub_y : np.ndarray
            Подвыборка меток классов.
        node : dict
            Узел дерева, который будет заполнен информацией о разбиении.

        """

            # Критерий останова: все элементы одного класса
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # Критерий останова: максимальная глубина
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Критерий останова: недостаточно объектов для разбиения
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        categories_map_best = None  # Сохраняем categories_map для категориальных признаков

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {
                    key: clicks.get(key, 0) / count for key, count in counts.items()
                }
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {
                    category: i for i, category in enumerate(sorted_categories)
                }
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Некорректный тип признака")

            unique_values = np.unique(feature_vector)
            if len(unique_values) <= 1:
                continue

            thresholds_sorted, ginis_sorted, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                    categories_map_best = None
                elif feature_type == "categorical":
                    threshold_best = [k for k, v in categories_map.items() if v < threshold]
                    categories_map_best = categories_map  # Сохраняем для предсказания

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
            node["categories_map"] = None
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
            node["categories_map"] = categories_map_best  # Сохраняем для предсказания
        else:
            raise ValueError("Некорректный тип признака")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        if node["type"] == "terminal":
          return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            threshold = node["threshold"]
            if x[feature] < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            categories_map = node.get("categories_map", {})
            categories_split = node.get("categories_split", [])
            value = x[feature]

            if value in categories_split:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Некорректный тип признака")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
