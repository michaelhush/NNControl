from NN_Control_algorithm import *

## Main program
## An instance of controller class
LSTM_cont = LSTM_controller(epitime = 6.4, timestep= 0.1)
LSTM_cont.config_lstm_para(batch_size= 2, plant_state_size = 64, cont_state_size = 64, setout = [0.0,-1.0,0.0])
LSTM_cont.config_lstm()
LSTM_cont.config_noise(nF = 5.0)
LSTM_cont.initialize_variables()
iterations = 1

LSTM_cont.train_data(episodes = 2)
LSTM_cont.train_plant(epochs = 2)
LSTM_cont.train_cont(epochs = 2)
LSTM_cont.update_cont_plant_states()
try:
    for it in range (iterations):
        print("Iteration:" + str(it+1))
        LSTM_cont.train_data(episodes = 2)
        LSTM_cont.train_plant(epochs = 2)
        LSTM_cont.train_cont(epochs = 2)
        LSTM_cont.update_cont_plant_states()

except KeyboardInterrupt:
    print("Execution interupted. Generating plots before ending.")

LSTM_cont.save_trained_NN('test')

# comp_noise = LSTM_cont.gen_noise()
# comp_init_state = LSTM_cont.gen_init_state()
# plt.figure(1)
# LSTM_cont.plot_plant_dynamics(noise=comp_noise,init_state=comp_init_state)
# plt.figure(2)
# LSTM_cont.plot_cont_plant_dynamics(noise=comp_noise,init_state=comp_init_state)
# plt.figure(3)
# LSTM_cont.plot_plant_cost_hist()
# plt.figure(4)
# LSTM_cont.plot_cont_cost_hist()
# plt.show()

# print("Displaying plots")
# print("Noise:")
# print(comp_noise[0])
# print("Initial state:")
# print(comp_init_state)

LSTM_cont.sess.close()
