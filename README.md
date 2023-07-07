# Anomaly_in_malaria_surveillance_data
Anomaly detection in epidemiological malaria data. We detect unusual events in the reported case data especially flareup and drifts in case numbers.

To execute:

1. Open Command prompt 
2. Navigate to the folder containing the files
3. Run: streamlit run anomaly.py

## Abstract to the Published journal paper (https://www.mdpi.com/2227-9032/11/13/1896) 

Disease surveillance is used to monitor ongoing control activities, detect early outbreaks,
and inform intervention priorities and policies. However, data from disease surveillance that could be
used to support real-time decision-making remain largely underutilised. Using the Brazilian Amazon
malaria surveillance dataset as a case study, in this paper we explore the potential for unsupervised
anomaly detection machine learning techniques to discover signals of epidemiological interest. We
found that our models were able to provide an early indication of outbreak onset, outbreak peaks, and
change points in the proportion of positive malaria cases. Specifically, the sustained rise in malaria in
the Brazilian Amazon in 2016 was flagged by several models. We found that no single model detected
all anomalies across all health regions. Because of this, we provide the minimum number of machine
learning models top-k models) to maximise the number of anomalies detected across different health
regions. We discovered that the top three models that maximise the coverage of the number and types
of anomalies detected across the thirteen health regions are principal component analysis, stochastic
outlier selection, and the minimum covariance determinant. Anomaly detection is a potentially
valuable approach to discovering patterns of epidemiological importance when confronted with a
large volume of data across space and time. Our exploratory approach can be replicated for other
diseases and locations to inform monitoring, timely interventions, and actions towards the goal
of controlling endemic disease.
