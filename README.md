# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
-> Step1 :
We are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

-> Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
</br>
</br>

## POLICY IMPROVEMENT FUNCTION
#### Name : NAVEEN S
#### Register Number : 212222240070
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
#### Name : NAVEEN S
#### Register Number : 212222240070
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="668" height="136" alt="image" src="https://github.com/user-attachments/assets/babf45f7-b5a4-472a-9e33-26bb254d3fdc" />
<img width="614" height="43" alt="image" src="https://github.com/user-attachments/assets/d36d4f34-1ca9-4398-abe7-285ebe6fe351" />
<img width="449" height="145" alt="image" src="https://github.com/user-attachments/assets/b24c80d8-ce21-47e4-a33b-36391ffc3770" />





### 2. Policy, Value function and success rate for the Improved Policy
<img width="507" height="138" alt="image" src="https://github.com/user-attachments/assets/8dc5e294-0ba7-4131-a2c9-722a2575e71b" />
<img width="563" height="32" alt="image" src="https://github.com/user-attachments/assets/c1926160-9ce5-414b-a3e9-07701d2f21cb" />
<img width="524" height="145" alt="image" src="https://github.com/user-attachments/assets/6865e59d-4029-4cf2-bd56-2eda1b0b368c" />





### 3. Policy, Value function and success rate after policy iteration
<img width="590" height="166" alt="image" src="https://github.com/user-attachments/assets/6325659c-4a57-4a35-8e92-37aa2138d933" />
<img width="591" height="19" alt="image" src="https://github.com/user-attachments/assets/83461bfa-2a07-4a14-92c7-9861bedcb438" />
<img width="746" height="113" alt="image" src="https://github.com/user-attachments/assets/224cda69-6861-41eb-a8d2-4e8f16aa157f" />



## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
