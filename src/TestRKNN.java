import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/**
 * Created by pablonsilva on 31/01/2018.
 */
public class TestRKNN {
    public static void main(String[] args)
    {
        String path = "/Users/pablonsilva/Google Drive/Doutorado/Biology of Ageing/Data/" +
                "GenAgev17/gene_ontology_mix/Threshold/3/MM-BP-threshold-3.arff";
        FileReader fr = null;
        Instances data = null;
        try {
            fr = new FileReader(path);
            BufferedReader br = new BufferedReader(fr);
            data = new Instances(br);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        data.setClassIndex(data.numAttributes()-1);

        int[] ind = {1,10,100,250,500,1000,2000};
        for(int i =0; i < ind.length; i++) {
            RandomKNN r_knn = new RandomKNN(ind[i], 1);
            try {
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(r_knn, data, 10, new Random(1));
                System.out.println(ind[i] + " - " + 100 * Math.sqrt(eval.truePositiveRate(1) * eval.trueNegativeRate(1)));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
