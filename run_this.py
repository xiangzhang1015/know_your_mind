from env_spatial_autoregressive import Env
import pickle
from RL_brain import DeepQNetwork

def run_env(best_reward=0, best_state=[], reward_his =[]):
    # start training
    MAX_EPISODES = 50
    MAX_EP_STEPS = 50
    for i in range(MAX_EPISODES):
        # print
        # initial state
        state = env.reset()
        # print 'initial state', state, state[0][1]
        for step in range(MAX_EP_STEPS):
            # env.render()
            action = RL.choose_action(state) #
            print 'episode', i, 'step', step, 'Action', action, 'state', state

            state_, reward, done, acc, indices = env.step(action, step, i)
            print 'state_, reward, done, acc',state_, reward, done, acc
            # store the state, action, reward and the next state
            # print state.shape, action.shape, reward.shape, state_.shape
            RL.store_transition(state, action, reward, indices, state_)
            reward_his.append([i, step, reward])
            # record best reward
            current_reward = reward

            if current_reward > best_reward:
                best_reward = reward
                best_state = [acc, reward, state, state_, indices]
            # print 'Do you want to build a snowman?'
            if (step > 200) and (step % 5 == 0):  # learn once for each 5 steps
                RL.learn()
            # update the state

            if done or step == MAX_EP_STEPS-1:
                print('This episode is done, start the next episode')
                break
            state = state_
    return best_reward, best_state, reward_his

if __name__ == "__main__":
    len_max = 128
    env = Env(len_max=len_max, n_fe=14, n_classes=6)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.8,
                      e_greedy=0.8,
                      replace_target_iter=200,
                      len_max=len_max,
                      memory_size=2000,
                      e_greedy_increment=0.002 # each step, the e_greedy will increase 0.002
                      # output_graph=True
                      )
    # env.after(10, run_env)0.01
    best_reward, best_state, reward_his = run_env()
    print best_state, best_reward
    # env.mainloop()
    # RL.plot_cost()
    pickle.dump(RL.cost_his, open("cost_his_emotiv", "wb"))
    pickle.dump(reward_his, open("reward_his_emotiv", "wb"))
    pickle.dump(best_state, open("best_state_emotiv", 'wb'))