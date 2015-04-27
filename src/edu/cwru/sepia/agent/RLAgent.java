package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;

import java.io.*;
import java.util.*;

public class RLAgent extends Agent {

    /**
     * Set in the constructor. Defines how many learning episodes your agent should run for.
     * When starting an episode. If the count is greater than this value print a message
     * and call sys.exit(0)
     */
    public final int numEpisodes;

    /**
     * List of your footmen and your enemies footmen
     */
    private List<Integer> myFootmen;
    private List<Integer> enemyFootmen;
    
    //Hashmap to store the rewards for each footmen; the integer is the unitID, and the double is the reward
    private HashMap<Integer, Double> footRewards = new HashMap<Integer, Double>();
    //private HashMap<Integer, Double> enemyRewards = new HashMap<Integer, Double>();
    
    //HashMap to store the previous Q values for each footmen
    private HashMap<Integer, Double> prevQValues = new HashMap<Integer, Double>();
    
    //HashMap to store the previous feature vector
    private HashMap<Integer, double[]> prevFeatures = new HashMap<Integer, double[]>();
    
    //double to store the total cumulative reward (undiscounted)
    private double cumulativeReward;
    
    private boolean runningEvals = false;
    private int numEvals = 0;
    private double evalReward = 0;
    private List<Double> avgRewards = new ArrayList<Double>();
    //An int to keep track the number of episodes completed so far
    private int numEpisodesCompleted = 0;
    
    private boolean shouldLoadWeights = false;
    //A counter representing the exponential component of the gamma factor
    int turnCounter = 1;
    /**
     * Convenience variable specifying enemy agent number. Use this whenever referring
     * to the enemy agent. We will make sure it is set to the proper number when testing your code.
     */
    public static final int ENEMY_PLAYERNUM = 1;

    /**
     * Set this to whatever size your feature vector is.
     */
    public static final int NUM_FEATURES = 3;

    /** Use this random number generator for your epsilon exploration. When you submit we will
     * change this seed so make sure that your agent works for more than the default seed.
     */
    public final Random random = new Random(12345);

    /**
     * Your Q-function weights.
     */
    public Double[] weights;
    
    /**
     * These variables are set for you according to the assignment definition. You can change them,
     * but it is not recommended. If you do change them please let us know and explain your reasoning for
     * changing them.
     */
    public final double gamma = 0.9;
    public final double learningRate = .0001;
    public final double epsilon = .02;

    public RLAgent(int playernum, String[] args) {
        super(playernum);

        if (args.length >= 1) {
            numEpisodes = Integer.parseInt(args[0]);
            System.out.println("Running " + numEpisodes + " episodes.");
        } else {
            numEpisodes = 10;
            System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
        }

        boolean loadWeights = false;
        if (args.length >= 2) {
            loadWeights = Boolean.parseBoolean(args[1]);
            System.out.println(loadWeights);
        } else {
            System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
        }

        if (loadWeights) {
            weights = loadWeights();
        } else {
            // initialize weights to random values between -1 and 1
            weights = new Double[NUM_FEATURES];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }
    }

    /**
     * We've implemented some setup code for your convenience. Change what you need to.
     */
    //TODO
    @Override
    public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {
    	
    	//Check if we ran 10 episodes already, if so, run evaluations
    	if(numEpisodesCompleted % 10 == 0 && !runningEvals) {
    		runningEvals = true;
    		numEvals = 0;
    	}
    	if(runningEvals && numEvals >= 5) {
    		runningEvals = false;
    	}
        // You will need to add code to check if you are in a testing or learning episode
    	cumulativeReward = 0;
        // Find all of your units
        myFootmen = new LinkedList<>();
        for (Integer unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                myFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        // Find all of the enemy units
        enemyFootmen = new LinkedList<>();
        for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                enemyFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }
        return middleStep(stateView, historyView);
    }

    /**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an even whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {
    	HashMap<Integer, Action> actions = new HashMap<Integer, Action>();
    	if (stateView.getTurnNumber() == 0) {
    		//Intialize the footRewards hashmap 
    		for(int footID : myFootmen) {
    			footRewards.put(footID, 0.0);
    		}
    		//First turn, so just randomly pick an action
    		for (int unit : myFootmen) {
    			int enemyID = selectAction(stateView, historyView, unit);
    			prevQValues.put(unit, calcQValue(stateView, historyView, unit, enemyID));
    			prevFeatures.put(unit, calculateFeatureVector(stateView, historyView, unit, enemyID));
    			actions.put(unit, Action.createCompoundAttack(unit, enemyID));
    		}
    	}
    	turnCounter++;
    	
    	
    	
    	
    	
    	boolean update = false;
    	
    	Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
    	//Check if a unit has died, and if so, remove that unit from the corresponding unit list
    	if(stateView.getTurnNumber() > 1) { 
    		//Iterate over all the footmen, and update their cumulative reward values
        	for (int unitID : myFootmen) {
        		cumulativeReward += calculateReward(stateView, historyView, unitID);
        		double tempReward = footRewards.get(unitID) + Math.pow(gamma, turnCounter) * calculateReward(stateView, historyView, unitID);
        		footRewards.put(unitID, tempReward);
        	}
        	
        	//Check if any unit has died, if so, set the update boolean to true
    		for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
    			if(enemyFootmen.contains(deathLog.getDeadUnitID())){
    				for (int i = 0; i < enemyFootmen.size(); i++) {
    					if(enemyFootmen.get(i) == deathLog.getDeadUnitID())  {
    						enemyFootmen.remove(i);
    						update = true;
    					}	
    				}
    			}
    			if(myFootmen.contains(deathLog.getDeadUnitID())){
    				for (int i = 0; i < myFootmen.size(); i++) {
    					if(myFootmen.get(i) == deathLog.getDeadUnitID()) {
    						myFootmen.remove(i);
    						update = true;
    					}
    				}
    			}
    		}
    		//If any footmen has taken damage, update the weights
    		for(int footmanId : myFootmen) {
    			for(DamageLog damageLog : historyView.getDamageLogs(stateView.getTurnNumber() - 1)){
    				if(footmanId == damageLog.getDefenderID()){
    					update = true;
    				}
    			}
    		}

        	
    		for (ActionResult result : actionResults.values()) {
            	if(result.getFeedback().equals(ActionFeedback.FAILED)) {
            		//The action failed, so update and reassign actions
            		//int unitId = result.getAction().getUnitId();
            		update = true;
            	}
            	if(result.getFeedback().equals(ActionFeedback.COMPLETED)) {
            		//The action was completed
            		update = true;
            	}
            }
    	}
    	
    	if (update) {
    		turnCounter = 0;
    		int enemyId;
    		double[] myWeights = new double[weights.length];
    		for(int i = 0; i < weights.length; i++) {
    			myWeights[i] = weights[i];
    		}
    		if(!runningEvals) {
    			for (int unitID : myFootmen) { 
    				//Update the weight using the stored weights, feature values (for each footmen), and rewards (stored for each footmen)
    				double[] tempWeights = updateWeights(myWeights, prevFeatures.get(unitID), footRewards.get(unitID), stateView, historyView, unitID);
    				//Populate the weights array with the new weight values
    				for(int i = 0; i < tempWeights.length; i++) {
    					weights[i] = tempWeights[i];
    				}

    				/*Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, stateView.getTurnNumber() - 1);
    	        for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
    	            if(commandEntry.getKey() == unitID){
    	            	//make sure this gets the ID of the attacked enemy
    	            	enemyId = commandEntry.getValue().getUnitId();
    	            }
    	        	System.out.println("Unit " + commandEntry.getKey() + " was commanded to " + commandEntry.getValue().toString());
    	        }*/
    			}
    		}
    		//After updating, iterate through all the footmen and select new actions for each of them
    		for (int unit : myFootmen) {

    			int enemyID = selectAction(stateView, historyView, unit);
    			prevQValues.put(unit, calcQValue(stateView, historyView, unit, enemyID));
    			prevFeatures.put(unit, calculateFeatureVector(stateView, historyView, unit, enemyID));
    			actions.put(unit, Action.createCompoundAttack(unit, selectAction(stateView, historyView, unit)));
    		}
    		//After updating, zero out the rewards
    		for(int footID : myFootmen) {
    			footRewards.put(footID, 0.0);
    		}
    	}
    	
         	
    	return actions;
    }

    /**
     * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
     * finished a set of test episodes you will call out testEpisode.
     *
     * It is also a good idea to save your weights with the saveWeights function.
     */
    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {
    	//TODO
        // MAKE SURE YOU CALL printTestData after you finish a test episode.
    	//System.out.println(cumulativeReward);
    	//System.out.println(numEpisodesCompleted);
    	if(!runningEvals) {
    		numEpisodesCompleted++;
    	}
    	else {
    		evalReward += cumulativeReward;
    		numEvals++;
    		if(numEvals >= 5) {
    			//runningEvals = false;
    			evalReward = evalReward / 5.0;
    			avgRewards.add(evalReward);
    		}
    	}

    	if(numEpisodesCompleted >= numEpisodes) {
    		printTestData(avgRewards);
    		System.out.println("Completed all episodes! Exiting now");
    		System.exit(0);
    	}
    	//numEpisodesCompleted++;
        // Save your weights
        saveWeights(weights);

    }

    /**
     * Calculate the updated weights for this agent. 
     * @param oldWeights Weights prior to update
     * @param oldFeatures Features from (s,a)
     * @param totalReward Cumulative discounted reward for this footman.
     * @param stateView Current state of the game.
     * @param historyView History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return The updated weight vector.
     */
    public double[] updateWeights(double[] oldWeights, double[] oldFeatures, double totalReward, State.StateView stateView, History.HistoryView historyView, int footmanId) {
        //TODO: Maybe done?
    	//w <-- wi - alpha*(-(R(s,a) + gamma*max(Qw(s',a')) - Qw(s,a))*fi(s,a)
    	double [] w = new double[oldWeights.length];
    	for(int k = 0; k < oldWeights.length; k++) {
    		w[k] = oldWeights[k];
    	}
    	
    	double Qprime = -9999999;
    	for(int enemyID: enemyFootmen){
    		Qprime = Math.max(Qprime, calcQValue(stateView, historyView, footmanId, enemyID));
    	}
    	int lastTurnNum = stateView.getTurnNumber()-1;
    	int enemyId = 0;
    	Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNum);
        /*for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
            if(commandEntry.getKey() == footmanId){
            	//make sure this gets the ID of the attacked enemy
            	enemyId = commandEntry.getValue().getUnitId();
            }
        	System.out.println("Unit " + commandEntry.getKey() + " was commanded to " + commandEntry.getValue().toString());
        }*/
    	
    	for(int i = 0; i < oldWeights.length; i++)
    	{
    		w[i] = oldWeights[i] + (learningRate * ((totalReward + gamma * Qprime - prevQValues.get(footmanId))) * oldFeatures[i]);
    	}
    	return w;
    }

    /**
     * Given a footman and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     *
     * @param stateView Current state of the game
     * @param historyView The entire history of this episode
     * @param attackerId The footman that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {
    	/*if (stateView.getTurnNumber() == 0) {
    		return random.nextInt(enemyFootmen.size());
    	}*/
    	int maxEnemyID = -1;
    	double maxQValue = Integer.MIN_VALUE;

    	for(int enemyID : enemyFootmen) {
    		//Set the maxQValue to be the greatest q value calculated so far
    		if(calcQValue(stateView, historyView, attackerId, enemyID) > maxQValue) {
    			maxQValue = calcQValue(stateView, historyView, attackerId, enemyID);
    			maxEnemyID = enemyID;
    		}
    	}



        //Do the greedy selection based on the probability of 1 - epsilon
        if(random.nextFloat() <= 1.0 - epsilon) {

        	return maxEnemyID;
        }
    	//Else, take a random action
        else {
        	int nextId = random.nextInt(enemyFootmen.size());
        	//If the random int generated is equal to the greedy selection, just try again
        	while(stateView.getUnit(enemyFootmen.get(nextId)) == null)  
        	{
        		nextId = random.nextInt(enemyFootmen.size());
        	}
        	maxEnemyID = enemyFootmen.get(nextId);
        	return maxEnemyID;
        }
        
    }

    /**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
     * for the full list of rewards.
     *
     * Remember that you will need to discount this reward based on the timestep it is received on. See
     * the assignment description for more details.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *     "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param stateView The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param footmanId The footman ID you are looking for the reward from.
     * @return The current reward
     */
    public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {
        //TODO: Done?
    	double reward = 0;
    	
    	int lastTurnNum = stateView.getTurnNumber()-1;
    	//reward += Math.pow(gamma, lastTurnNum);
    	
    	//penalize for each command issued (for each action taken)
    	Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNum);
        for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
            if(commandEntry.getKey() == footmanId) {
            	reward -= 0.1;
            }
        }
    	
    	//calculate rewards based on damage given/taken
    	for(DamageLog damageLog : historyView.getDamageLogs(lastTurnNum)){
    		if(footmanId == damageLog.getAttackerID()){
    			reward += damageLog.getDamage();
    		}
    		else if(footmanId == damageLog.getDefenderID()){
    			reward -= damageLog.getDamage();
    		}
    	}
    	
    	//calculate rewards based on deaths
    	for(DeathLog deathLog : historyView.getDeathLogs(lastTurnNum)){
    		if(footmanId == deathLog.getDeadUnitID()){
    			reward -= 100;
    		}
    		else{
    			reward += 100;
    		}
    	}
    	
    	return reward;
    }

    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will calculate
     * your features and multiply them by your current weights to get the approximate Q-value.
     *
     * @param stateView Current SEPIA state
     * @param historyView Episode history up to this point in the game
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman that your footman would be attacking
     * @return The approximate Q-value
     */
    public double calcQValue(State.StateView stateView,
                             History.HistoryView historyView,
                             int attackerId,
                             int defenderId) {
    	//TODO
    	//Qw(s,a) = summation(wi * fi(s,a)
    	double Qw = 0;

    	double [] feature = calculateFeatureVector(stateView, historyView, attackerId, defenderId);
    	for(int i = 0; i < weights.length; i++){
    		Qw += weights[i] * feature[i];
    	}

        return Qw;
    }

    /**
     * Given a state and action calculate your features here. Please include a comment explaining what features
     * you chose and why you chose them.
     *
     * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
     * take a dot product of this array with the weights array to get a Q-value for a given state action.
     *
     * It is a good idea to make the first value in your array a constant. This just helps remove any offset
     * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
     * description.
     *
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The array of feature function outputs.
     */
    public double[] calculateFeatureVector(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {

    	double[] featureArr = new double[NUM_FEATURES];
    	featureArr[0] = 1.0;
    	
    	Unit.UnitView myUnit = stateView.getUnit(attackerId);
    	Unit.UnitView enemyUnit = stateView.getUnit(defenderId);
    	
    	//First feature is the inverse of the euclidean distance between the footmen and the enemy unit
    	featureArr[1] = 1.0 / (euclideanDistance(myUnit, enemyUnit) + 1.0);
    	//featureArr[1] = euclideanDistance(myUnit, enemyUnit);
    	//Second feature is the ratio of their hp
    	featureArr[2] = (double)myUnit.getHP() / (double)enemyUnit.getHP();
    	
    	
    	//need to pick how we want to do features.
    	//Possible ones: Distance between f & e; how many f attacking that e; is e attacking f?

        return featureArr;
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning rate data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    public void printTestData (List<Double> averageRewards) {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++) {
            String gamesPlayed = Integer.toString(10*i);
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++) {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will take your set of weights and save them to a file. Overwriting whatever file is
     * currently there. You will use this when training your agents. You will include th output of this function
     * from your trained agent with your submission.
     *
     * Look in the agent_weights folder for the output.
     *
     * @param weights Array of weights
     */
    public void saveWeights(Double[] weights) {
        File path = new File("agent_weights/weights.txt");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
     * can be created using the saveWeights function. You will use this function if the load weights argument
     * of the agent is set to 1.
     *
     * @return The array of weights
     */
    public Double[] loadWeights() {
        File path = new File("agent_weights/weights.txt");
        if (!path.exists()) {
            System.err.println("Failed to load weights. File does not exist");
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            List<Double> weights = new LinkedList<>();
            while((line = reader.readLine()) != null) {
                weights.add(Double.parseDouble(line));
            }
            reader.close();

            return weights.toArray(new Double[weights.size()]);
        } catch(IOException ex) {
            System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
        }
        return null;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
    
    public int chebyshevDistance(int unit1x, int unit1y, int unit2x, int unit2y) {
        return Math.max(Math.abs(unit1x - unit2x), Math.abs(unit1y - unit2y));
    }
    
    public double euclideanDistance(UnitView myUnit, UnitView enemyUnit) {
        return Math.sqrt(Math.pow(myUnit.getXPosition() - enemyUnit.getXPosition(), 2) + Math.pow(myUnit.getYPosition() - enemyUnit.getYPosition(), 2));
    }
}
