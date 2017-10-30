from processor import Processor
from Analytics import Analytics


def project():
	pr = Processor()

	pr.read_data('presidential_polls.csv')
	pr.draw_visualization()
	new_df = pr.clean_prepare_data()
	an = Analytics(new_df)

	X, y, X_train, X_test, y_train, y_test = an.test_train_data()
	log_model, logistic_model_pred, logistic_model_prob = an.logistic_model(
	                                                                     X_train,
	                                                                     y_train,
	                                                                     X_test)
	desc_model, decision_model_pred, decision_model_prob = an.decision_tree_model(
		X_train, y_train, X_test)
	an.cross_validation(log_model, X, y, desc_model)

	an.accuracy(y_test, logistic_model_pred, logistic_model_prob,
	         decision_model_pred, decision_model_prob)


if __name__ == "__main__":
	project()
