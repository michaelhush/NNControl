from NN_Control_algorithm import *

LSTM_cont2 = LSTM_controller(epitime = 6.4, timestep= 0.1)
LSTM_cont2.config_lstm_para(batch_size= 2, plant_state_size = 64, cont_state_size = 64, setout = [0.0,-1.0,0.0])
LSTM_cont2.load_trained_NN('test')
LSTM_cont2.train_data(episodes = 2)
LSTM_cont2.train_plant(epochs = 2)
LSTM_cont2.train_cont(epochs = 2)
comp_noise = LSTM_cont2.gen_noise()
comp_init_state = LSTM_cont2.gen_init_state()
plt.figure(1)
LSTM_cont2.plot_plant_dynamics(noise=comp_noise,init_state=comp_init_state)
plt.show()