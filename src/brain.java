import java.util.ArrayList;

public class brain {
    private int[] neural_type;
    private ArrayList<Double[]> dataset;
    private ArrayList<Double[]> desired_data;
    private int maxEpoch;
    private double minError;

    private double  error_n; //sum of squared error at iteration n (sse)
    private double avg_error_n = 1 ; // average sse of all epoch
    private double biases = 1; // threshold connected

    private weightMatrix[] layer_weight  ;
    private double[][]  node  ;


    public brain(String _neural_type ,ArrayList<Double[]> _dataset,ArrayList<Double[]> _desired_data  , int _maxEpoch , double _minError){
        String[] splitArray = _neural_type.split("");
        int[] array = new int[splitArray.length];

        for (int i = 0; i < splitArray.length; i++) {
            array[i] = Integer.parseInt(splitArray[i]);

        }
        this.neural_type = array;
        init_Structor();
        this.dataset = _dataset;
        this.desired_data = _desired_data;
        this.maxEpoch = _maxEpoch;
        this.minError = _minError;


    }
    private void init_Structor(){
        node = new double[neural_type.length][];
        for (int i = 0; i < neural_type.length; i++) {
            node[i] = new double[neural_type[i]];
        }
        layer_weight = new weightMatrix[neural_type.length-1];
        for (int layer = 0; layer < layer_weight.length; layer++) {
            weightMatrix weight = new weightMatrix(neural_type[layer+1],neural_type[layer] );
            layer_weight[layer] = weight;
        }

    }
    public void train(){
        int N =0;
        while (N < maxEpoch && avg_error_n > minError){
            int dataset_i = (int) (Math.random() * ((dataset.size()) + 1));
            for (int i = 0; i < neural_type.length; i++) { // each col



                for (int j = 0; j < neural_type[i]; j++) { // each row

//                    double suminput = sum(i , j);
//                    double node_output = actication_fn(suminput);
//                    node[i][j] = node_output;

                }
            }

            N++;
        }
        save_weight();
    }

    private void save_weight() {
        //TODO
    }

//    public double sum(int i , int j ){
//        double sum = 0;
//        for (double y: node[i-1]) {
//            sum += y*weight[j][i];
//        }
//        return sum + biases;
//
//    }
    public double actication_fn(Double x){
        return 1 / (1 + Math.exp(-x));
    }
}
