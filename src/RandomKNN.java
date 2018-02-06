import weka.classifiers.AbstractClassifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Random;

/**
 * How does RandomKNN work?
 * 1 - A collection of r different kNN classifier is generated, each one takes a random subset
 * of the input variables.
 * 2 - Each KNN classifier classifies a test point by its majority, or weighted majority class,
 * of its k nearest neighbors.
 * 3 - The final classification in each case is determined by majority voting of RKNN classifications.
 * Created by pablonsilva on 31/01/2018.
 */
public class RandomKNN extends AbstractClassifier {

    private Random _rand;
    private IBk[] _knn_collection;
    private ArrayList<int[]> _ind_array_collection; // we need this in the distributionForInstance method,
    //we have to remove features from an unseen instance to make a classification properly.

    public RandomKNN(int r, int k)
    {
        this(r,k,new Random(1));
    }

    public RandomKNN(int r, int k, Random rand)
    {
        _rand = rand;
        _ind_array_collection = new ArrayList<>();
        _knn_collection = new IBk[r];
        for(int i = 0; i < _knn_collection.length;i++) {
            _knn_collection[i] = new IBk(k);
        }
    }

    @Override
    public void buildClassifier(Instances data) throws Exception
    {
        for(int i = 0 ; i < _knn_collection.length; i++)
        {
            int[] ind = randomSubsetOfFeatures(data);
            _ind_array_collection.add(i,ind);
            Instances partialTrainingData = arrayToInstances(ind,data);
            _knn_collection[i].buildClassifier(partialTrainingData);
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        double[] dist_total = new double[instance.numClasses()];

        for (int i = 0; i < _knn_collection.length; i++){
            Instances dataAux = new Instances(instance.dataset(),0);
            dataAux.clear();
            dataAux.add(instance);

            Instances dataRmv = arrayToInstances(_ind_array_collection.get(i),dataAux);
            //double[] dist = _knn_collection[i].distributionForInstance(dataRmv.firstInstance());

//            for( int h = 0 ; h < dist.length; h++)
//            {
//                dist_total[h] += dist[h];
//            }

            dist_total[(int)_knn_collection[i].classifyInstance(dataRmv.firstInstance())] += 1.0/_knn_collection.length;
        }

//        for( int h = 0 ; h < dist_total.length; h++)
//            dist_total[h] /= (double)_knn_collection.length;


        return dist_total;
    }

    private Instances arrayToInstances(int[] featuresToKeep, Instances data)
    {
        Instances newData = new Instances(data);
        Remove rmv = null;
        Instances fData = null;
        try {
            // These methods must be called exactly in this order.
            rmv = new Remove();
            rmv.setAttributeIndicesArray(featuresToKeep);
            rmv.setInvertSelection(true);
            rmv.setInputFormat(data);

            fData  = Filter.useFilter(newData,rmv);
        } catch (Exception e)
        {
        }

        return fData;
    }


    private int[] randomSubsetOfFeatures(Instances data) {
        // array with selected features
        boolean[] featuresSelected = new boolean[data.numAttributes() - 1];
        int count_selected_features = 0;
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            if (_rand.nextDouble() > 0.5) {
                featuresSelected[i] = true;
                count_selected_features++;
            }
        }

        int[] featuresToKeep = new int[count_selected_features+1];
        int c = 0;
        for(int i = 0 ; i < data.numAttributes()-1; i++)
        {
            if(featuresSelected[i]) {
                featuresToKeep[c] = i;
                c++;
            }
        }

        //save the class attribute
        featuresToKeep[featuresToKeep.length-1] = data.classIndex();
        return  featuresToKeep;
    }
}
