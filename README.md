The idea behind this project was to create a dashboard where one could monitor equipment, in this case I chose to simulate an HVAC system but this can be tailored to different needs. Think of motors, compressors, pumping stations, vacuums etc. 

I tested using an isolation forest (ML) which is an unsupervised learning algorithm for anomaly detection that isolates anomalies instead of profiling it. The model learns what the normal is and learns it, and it scores new readings on how anomalous they are (0 = normal, 1 = anomaly).

In a real world scenario, a dashboard like this can be paired to live sensors. Which in turn can be tied to a messaging system such as a Teams Bot, WhatsApp, Telegram or a similar service to allow team escallation. Finally, the data can be stored for future reference and troubleshooting activities. 

