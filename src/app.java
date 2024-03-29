
import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author anthony tong
 * @version 1.0
 */
public class app {
	private static int numRows = 846;
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 18, hiddenLayer = 140, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) throws IOException {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }
    
    private static Integer countRows() throws IOException {
    	int rows = 0;
    	LinkedList<String> filnames= new LinkedList<>();
    	filnames.add("xaa.dat");
    	filnames.add("xab.dat");
    	filnames.add("xac.dat");
    	filnames.add("xad.dat");
    	filnames.add("xae.dat");
    	filnames.add("xaf.dat");
    	filnames.add("xag.dat");
    	filnames.add("xah.dat");
    	filnames.add("xai.dat");
    	for (int i = 0; i<filnames.size();i++) {
    		String fil = String.format("data/vehicle_silouettes/%s", (String)filnames.get(i));
            BufferedReader br = new BufferedReader(new FileReader(new File(fil)));
            while (br.readLine() != null) rows++;
    	}
    	return rows;
    }

    private static Instance[] initializeInstances() {

//        double[][][] attributes = new double[4177][][];
//        double[][][] attributes = new double[12330][][];
        double[][][] attributes = new double[846][][];
        LinkedList<String> filnames= new LinkedList<>();
    	filnames.add("xaa.dat");
    	filnames.add("xab.dat");
    	filnames.add("xac.dat");
    	filnames.add("xad.dat");
    	filnames.add("xae.dat");
    	filnames.add("xaf.dat");
    	filnames.add("xag.dat");
    	filnames.add("xah.dat");
    	filnames.add("xai.dat");
        try {
    		String fil = String.format("data/vehicle_silouettes/%s", (String)filnames.pop());
            BufferedReader br = new BufferedReader(new FileReader(new File(fil)));
//            Scanner scan = new Scanner(br.readLine());

            for(int i = 0; i < attributes.length; i++) {
            	String line = br.readLine();
            	if (line == null) {
            		fil = String.format("data/vehicle_silouettes/%s", (String)filnames.pop());
            		br = new BufferedReader(new FileReader(new File(fil)));
            		line = br.readLine();
            	}
                Scanner scan = new Scanner(line);
                scan.useDelimiter(" ");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[18]; // 18 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 18; j++) {
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                }
                HashMap<String,Double> classifications = new HashMap();
                classifications.put("saab", 1.0);
                classifications.put("bus", 2.0);
                classifications.put("van", 3.0);
                classifications.put("opel", 4.0);
                //convert type to double
                // ['saab','bus','van','opel']
                String temp = scan.next();
//                System.out.println(classifications.get(temp));
                attributes[i][1][0] = classifications.get(temp);
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
//        	System.out.println(attributes[i][0]);
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0] < 15 ? 0 : 1));
        }

        return instances;
    }
}
