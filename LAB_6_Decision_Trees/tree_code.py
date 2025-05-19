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

    # Преобразование в числовой тип
    feature_vector = np.array(feature_vector, dtype=float)
    target_vector = np.array(target_vector, dtype=int)

    # Удаляем NaN значения
    valid_mask = ~np.isnan(feature_vector)
    feature_vector = feature_vector[valid_mask]
    target_vector = target_vector[valid_mask]

    if len(feature_vector) == 0:
        return np.array([]), np.array([]), None, None

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    unique_features = np.unique(sorted_features)

    if len(unique_features) <= 1:
        return np.array([]), np.array([]), None, None

    thresholds_candidates = (unique_features[:-1] + unique_features[1:]) / 2

    thresholds_list = []
    ginis_list = []
    best_gini = -np.inf
    best_threshold = None

    total_size = len(target_vector)

    for threshold in thresholds_candidates:
        left_size = np.searchsorted(sorted_features, threshold, side='left')
        right_size = total_size - left_size

        if left_size == 0 or right_size == 0:
            continue

        thresholds_list.append(threshold)

        sum_left = np.sum(sorted_targets[:left_size])
        sum_right = np.sum(sorted_targets[left_size:])

        # Энтропия для левого поддерева
        if left_size == 0:
            H_left = 0
        else:
            p1_left = sum_left / left_size
            p0_left = 1 - p1_left
            H_left = 1 - p1_left**2 - p0_left**2

        # Энтропия для правого поддерева
        if right_size == 0:
            H_right = 0
        else:
            p1_right = sum_right / right_size
            p0_right = 1 - p1_right
            H_right = 1 - p1_right**2 - p0_right**2

        # Критерий Джини
        gini = - (left_size / total_size) * H_left - (right_size / total_size) * H_right
        ginis_list.append(gini)

        if gini > best_gini or (gini == best_gini and threshold < best_threshold):
            best_gini = gini
            best_threshold = threshold

    if not thresholds_list:
        return np.array([]), np.array([]), None, None

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

    def _fit_node(self, sub_X, sub_y, node, depth=0):  
        """
        Обучение узла дерева решений с учетом критериев останова.
        
        Parameters
        ----------
        sub_X : np.ndarray
            Подвыборка признаков.
        sub_y : np.ndarray
            Подвыборка меток классов.
        node : dict
            Узел дерева, который будет заполнен информацией о разбиении.
        depth : int, optional
            Текущая глубина узла (по умолчанию 0).
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

        # Критерий останова: недостаточно образцов для разбиения
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        categories_map_best = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / count for key, count in counts.items()}
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {category: i for i, category in enumerate(sorted_categories)}
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Некорректный тип признака")

            if len(np.unique(feature_vector)) <= 1:
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
                    categories_map_best = categories_map

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
            node["categories_map"] = categories_map_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node.get("categories_split", []):
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Некорректный тип признака")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
