/*
 * @author anthony tong
 * */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;

import func.nn.backprop.RPROPUpdateRule;
import func.nn.backprop.WeightUpdateRule;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.activation.ActivationFunction;
import func.nn.activation.DifferentiableActivationFunction;
import func.nn.activation.HyperbolicTangentSigmoid;
import func.nn.activation.RELU;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.GradientErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.writer.CSVWriter;

public class Backpropexp{
	

    private static Instance[] training_instance = initializeInstances("data/vehicle_silouettes/train.dat",490);
    private static Instance[] test_instance = initializeInstances("data/vehicle_silouettes/test.dat",178);
    private static Instance[] cv_instance = initializeInstances("data/vehicle_silouettes/cv.dat",178);

    private static int inputLayer = 18, hiddenLayer = 140, outputLayer = 4, trainingIterations = 5000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet trainingset = new DataSet(training_instance);
    private static DataSet testset = new DataSet(test_instance);
    private static DataSet cvset = new DataSet(cv_instance);
//    private static DifferentiableActivationFunction act = new HyperbolicTangentSigmoid();//RELU();
    private static DifferentiableActivationFunction act = new RELU();
    
    public static void main(String[] args) throws IOException {
    	WeightUpdateRule rule = new RPROPUpdateRule(0.0033, 50, 0.000001);
    	BackPropagationNetwork classificationNetwork = factory.createClassificationNetwork(
    			new int[] {inputLayer, hiddenLayer, outputLayer},act);
    	NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainingset,classificationNetwork,measure);
//    	train(ga,classificationNetwork, trainingIterations, training_instance, cv_instance);
		train(new BatchBackPropagationTrainer(trainingset,classificationNetwork, (GradientErrorMeasure )measure, rule),classificationNetwork, 
				1000, training_instance,cv_instance);
    	for (int i=0;i<training_instance.length;i++) {
    		classificationNetwork.setInputValues(training_instance[i].getData());
    		classificationNetwork.run();
    	}
    }
    
    private static void train(BatchBackPropagationTrainer oa, BackPropagationNetwork network, int iterations,
    		Instance[] train_inst, Instance[] cv_inst) throws IOException {

		String[] headers= {"iterations","training score","cv score","training RMSE","cv RMSE"};
        CSVWriter writer = new CSVWriter("part2/backproprelu.csv",headers);
    	writer.open();
    	for (int i=0;i<iterations;i++) {
    		oa.train();
    		if (i%10==0) {
            	writer.nextRecord();
    	        writer.write(Integer.toString(i));
    			double trainscore= score(network, train_inst, measure);
    			double cvscore = score(network, cv_inst, measure);
    			double trainsquareerror= Helper.RMSE(network,train_inst);
    			double cvsquareerror= Helper.RMSE(network,cv_inst);
    	        writer.write(Double.toString(trainscore));
    	        writer.write(Double.toString(cvscore));
    	        writer.write(Double.toString(trainsquareerror));
    	        writer.write(Double.toString(cvsquareerror));
    		}
    	}
    	writer.close();
    }
    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, int iterations,
    		Instance[] train_inst, Instance[] cv_inst) {
    	for (int i=0;i<iterations;i++) {
    		oa.train();
    		if (i%10==0) {
    			score(network, train_inst, measure);
    			score(network, cv_inst, measure);
    		}
    	}
    }
    
    private static double score(BackPropagationNetwork network, Instance[] inst, ErrorMeasure measure) {
    	int correct=0,incorrect=0;
    	for (int i = 0; i<inst.length;i++) {
    		Instance instance = inst[i];
    		network.setInputValues(instance.getData());
    		network.run();
    		int pred=0, act=0;
    		double max=0;
    		for (int j = 0; j<4;j++) {
	    		double actual = instance.getLabel().getContinuous(j);
	    		double predicted = network.getOutputValues().get(j);
//	    		System.out.println(predicted);
	    		if (actual==1.0) {
//	    			System.out.println("actual: "+Integer.toString(j));
	    			act=j;
	    		}
	    		if (predicted>=max) {
	    			max=predicted;
//	    			System.out.println("predicted: "+Integer.toString(j));
	    			pred=j;
	    		}
	    		
    		}
//			System.out.println("predicted: "+Integer.toString(pred));
    		if(pred==act) {
    			correct++;
    		}else {
    			incorrect++;
    		}
    	}
//    	System.out.println("here");
//		System.out.println(correct/(float)(correct+incorrect));
    	return correct/(float)(correct+incorrect);
    }
    
    private static Instance[] initializeInstances(String filnam,int numrows) {
        double[][][] attributes = new double[numrows][][];
        try {
	        BufferedReader br = new BufferedReader(new FileReader(new File(filnam)));
	        for (int i = 0; i < attributes.length;i++) {
	        	String line = br.readLine();
	        	Scanner scan = new Scanner(line);
	            scan.useDelimiter(" ");
	
	            attributes[i] = new double[2][];
	            attributes[i][0] = new double[18]; // 18 attributes
	            attributes[i][1] = new double[1];
	            double[] mean= {94.80851064,45.18617021,83.31914894,171.2180851,61.60638298,8.64893617,
	            		171.3297872, 40.29787234,20.7606383,148.787234,190.712766,452.2606383,176.5425532,
	            		71.9893617,6.35106383,12.06914894,189.1968085,195.8617021};
	            for(int j = 0; j < 18; j++) {
	                attributes[i][0][j] = Double.parseDouble(scan.next())/mean[j];
	            }
	            HashMap<String,Double> classifications = new HashMap();
	            classifications.put("saab", 1.0);
	            classifications.put("bus", 2.0);
	            classifications.put("van", 3.0);
	            classifications.put("opel", 0.0);
	            String temp = scan.next();
	            attributes[i][1][0] = classifications.get(temp);
	            
	        }
	        Instance[] instances = new Instance[attributes.length];
	        for(int i = 0; i < instances.length; i++) {
//	        	System.out.println(attributes[i][0]);
	            instances[i] = new Instance(attributes[i][0]);
	            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
//	            instances[i].setLabel(new Instance(attributes[i][1][0] < 15 ? 0 : 1));
	            double[] classes = new double[4];
	            classes[(int)attributes[i][1][0]]=1.0;
	            instances[i].setLabel(new Instance(classes));
	        }

	        return instances;
        }
        catch (Exception e){
        	return null;
        }

    }
}