import queue
import time

from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC as SVC
from sklearn.tree import ExtraTreeClassifier

QUE = queue.Queue(4)
svc = SVC()
bayes = BernoulliNB()
ada = AdaBoostClassifier()
tree = ExtraTreeClassifier()
vote = VotingClassifier([('svc', svc), ('bayes', bayes), ('ada', ada), ('tree', tree)])


class Result:
    def __init__(self, cls, cls_name, y_true, y_pred):
        self.cls = cls
        self.cls_name = cls_name
        self.y_true = y_true
        self.y_pred = y_pred

    def print_results(self):
        print("Results for {} ".format(self.cls_name))
        print(classification_report(y_true=self.y_true, y_pred=self.y_pred))


def train_classifier(cls, X_train, y_train, cls_name):
    print("Start training {}".format(cls_name))
    cls.fit(X_train, y_train)
    print("{} is trained".format(cls_name))


def make_predictions(cls, X_test, cls_name, y_test):
    print("Start making predictions for {}".format(cls_name))
    y_pred = cls.predict(X_test)
    print("Finish predicting for {}".format(cls_name))
    result = Result(cls, cls_name, y_test, y_pred)
    QUE.put(result, block=True)


def compute_bayes():
    cls, cls_name = bayes, "Naive Bayes"
    train_classifier(cls, X_train, y_train, cls_name)
    make_predictions(cls, X_test, cls_name, y_test)


def compute_ada():
    cls, cls_name = ada, "Ada Boost"
    train_classifier(cls, X_train, y_train, cls_name)
    make_predictions(cls, X_test, cls_name, y_test)


def compute_svc():
    cls, cls_name = svc, "Support Vector Machines"
    train_classifier(cls, X_train, y_train, cls_name)
    make_predictions(cls, X_test, cls_name, y_test)


def compute_tree():
    cls, cls_name = tree, "Extra Tree"
    train_classifier(cls, X_train, y_train, cls_name)
    make_predictions(cls, X_test, cls_name, y_test)


def load_data():
    return datasets.fetch_20newsgroups_vectorized(subset="train").data, \
           datasets.fetch_20newsgroups_vectorized(subset="test").data, \
           datasets.fetch_20newsgroups_vectorized(subset="train").target, \
           datasets.fetch_20newsgroups_vectorized(subset="test").target


X_train, X_test, y_train, y_test = load_data()
if __name__ == '__main__':

    compute_bayes()
    compute_ada()
    compute_svc()
    compute_tree()
    train_classifier(vote, X_train, y_train, "Vote classifier")
    vote_predicts = vote.predict(X_test)
    finish = time.time()
    print("Vote results: ")
    print(classification_report(y_test, vote_predicts))
    while QUE.qsize() != 0:
        QUE.get_nowait().print_results()
