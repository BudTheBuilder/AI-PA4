Jiawei He, jxh602
Selena Pigoni, scp44

Our .zip contains the following files:
RLAgent.java
saved weights
This README

RLAgent.java Notes:

We are storing the discounted rewards for each footmen in a HashMap, as well as the features for each footmen, and the previous state’s Q values. These will be used in the when we are updating weights (since they require the previous state’s features and Q values)

* initialStep()
	* Determines whether it's a test episode or a learning episode
* middleStep()
	* records previous Q values and previous features
	* gets new cumulative
	* updates the weights
	* gets new actions for each footman
* terminalStep()
	* checks if all episodes are completed.
	* Prints average cumulative rewards
	* Saves weights
* updateWeights() updates the weight values
	* uses equation: (w <-- wi - alpha*(-(R(s,a) + gamma*max(Qw(s',a')) - Qw(s,a))*fi(s,a))
* selectAction(): for each footman, selects the action that gives the greatest Q function
	* returns the enemyId to attack
* calculateRewards() calculates the rewards for the previous turn
	* -.1 for each command
	* +d for damage against enemies, -d for damage against our footmen
	* +100 for enemy death, -100 for our footman death
* calcQValue() calculates the Q value for a given state and action.
	* uses equation: (Qw(s,a) = summation(wi * fi(s,a)))
	* uses calculateFeatureVector()
* calculateFeatureVector() calculates the value of the feature values
	* feature values include
		* distance between a footman and an enemy (better to attack nearby enemies)
		* ratio of footman health to enemy health (better to attack enemies weaker than you)