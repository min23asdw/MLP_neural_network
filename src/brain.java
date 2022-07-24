import java.util.ArrayList;

public class brain {
    private int[] neural_type;
    private ArrayList<Double[]> train_dataset;
    private ArrayList<Double[]> train_desired_data;


    private ArrayList<Double[]> test_dataset;
    private ArrayList<Double[]> test_desired_data;

   
    private int maxEpoch;
    private double minError;
    private double learning_rate;

    private ArrayList<Double[]> error_n = new ArrayList<>(); //sum of squared error at iteration n (sse)
    private double avg_error_n = 1 ; // average sse of all epoch
    private double biases = 0; // threshold connected : biases

    private Matrix[] layer_weight  ;
    private Double[][]  node  ;
    private Double[][]  local_gradient_node  ;


    public brain(String _neural_type ,ArrayList<Double[]> _train_dataset,ArrayList<Double[]> _train_desired_data  , int _maxEpoch , double _minError , double  _learning_rate){

        String[] splitArray = _neural_type.split("");
        int[] array = new int[splitArray.length];
        for (int i = 0; i < splitArray.length; i++) array[i] = Integer.parseInt(splitArray[i]);

        this.neural_type = array;

        init_Structor();

        this.train_dataset = _train_dataset;
        this.train_desired_data = _train_desired_data;
        this.maxEpoch = _maxEpoch;
        this.minError = _minError;
        this.learning_rate = _learning_rate;


    }
    private void init_Structor(){
        node = new Double[neural_type.length][];
        local_gradient_node = new Double[neural_type.length][];
        for (int i = 0; i < neural_type.length; i++) {
            node[i] = new Double[neural_type[i]];
            local_gradient_node[i] = new Double[neural_type[i]];
        }

        layer_weight = new Matrix[neural_type.length-1];
        for (int layer = 0; layer < layer_weight.length; layer++) {
            Matrix weight = new Matrix(neural_type[layer+1],neural_type[layer] ,true);
            layer_weight[layer] = weight;
        }

    }

    public void train(){
        int N =0;
        while (N < maxEpoch){ //&& avg_error_n > minError){

            //setup input node
//            int ran_dataset_i = (int) (Math.random() * ((train_dataset.size()) ));
            int ran_dataset_i = 256; // debug


            //set dataset value to input node
            for(int input_i = 0 ; input_i < neural_type[0] ; input_i ++){
                node[0][input_i] = train_dataset.get(ran_dataset_i)[input_i];
            }

            //cal ∑(input x weight) -> activation_Fn  for each neuron_node
            for(int layer = 0; layer < neural_type.length-1 ; layer++){
                node_eval(layer);
            }


            // ∑E(n) = 1/2 ∑ e^2 n  : sum of squared error at iteration n (sse)
            int number_outputn_node  =   node[node.length-1].length;
            Double[] errors = new Double[number_outputn_node];
            for ( int outnode_j = 0 ; outnode_j < number_outputn_node ; outnode_j++) {
                //train_desired_data => d_j desired output for neuron_node j at iteration N // it have "one data"
                //e_j  = error at neuron j at iteration N
                //TODO
                int diff_fn = 1;

                errors[outnode_j] =  train_desired_data.get(ran_dataset_i)[0] - node[node.length-1][outnode_j];
                local_gradient_node[node.length-1][outnode_j] =  errors[outnode_j] * diff_fn;
            }
            error_n.add(errors);


            // avg_E(n) = 1/N ∑ E(n)  : avg (sse)
            Double sum = 0.0;
            for (int i = 0 ; i < error_n.size() ; i++){
                 sum += error_n.get(i)[0];
            }
            avg_error_n =  sum / error_n.size();


            //⊃ sse/w_j =  ⊃ ∑(n) / ⊃ e_j  *  ⊃ e_j / ⊃ Y_j  *  ⊃ Y_j / ⊃ V_j  *  ⊃ V_j / ⊃ w_ji
            //⊃ sse/w_j =     (e_j(n))     *           -1    * diff Y_j(sum_input)  * Y_i

            // Y_j is  linear_fn
            //diff Y_j =  linear_fn -> 1
            //TODO
            // ɳ =  learning rate
            // ∆weight_ji = - ɳ (  ⊃ sse/w_j )
            // ∆weight_ji  =  ɳ [ (e_j(n)) * diff Y_j(sum_input) * Y_i ]



            //TODO
            // wji_next = wji_now + ∆weight_ji
            // output change_weight

            int output_layer = node.length-1;
            delta_weight_outputnode(output_layer);

            //local gradient output_k= e_k * diff Y_k
            // local gradient hidden_j = diff Y_j * ∑ (  W_kj  *  l_g k)

            for (int layer = layer_weight.length-1 ; layer >= 0  ; layer--) {
                //TODO
                int diff_fn = 1;

                double l_g = 0;
                for (int j = 0; j < node[layer].length   ; j++){
                    double sum_j = 0;
                    for(int k=0;k< node[layer+1].length  ; k++)
                    {
                        sum_j += layer_weight[layer].data[k][j] * local_gradient_node[layer+1][k] ;
                    }
                    l_g  += sum_j;
                    local_gradient_node[layer][j] = l_g * diff_fn;
                }
            }
            // hidden layer change_weight
            // ∆weight_ji =   ɳ *  local_gradient_j * Y_i

            for (int layer = node.length-2 ; layer > 0  ; layer--) {
                //TODO
                delta_weight_hiddennode(layer);

            }

            System.out.println(train_desired_data.get(ran_dataset_i)[0] + ":"+ node[node.length-1][0] + ":" + (train_desired_data.get(ran_dataset_i)[0] - node[node.length-1][0]) );

            N++; // next epoch
        }
        save_weight();
    }


    public void delta_weight_outputnode(int layer){
        //TODO it should to global?
        int weight_layer = layer-1;
        Matrix change_weight =  new Matrix(error_n.get(0).length, node[layer-1].length ,false);
        //TODO
        int diff_fn = 1;

        //mutiply matrix
        for (int j = 0; j < error_n.get(0).length ; j++){
            for(int i=0;i< node[layer-1].length ; i++)
            {
                double delta_weight = learning_rate * (error_n.get(0)[j] * diff_fn * node[layer-1][i] );
                change_weight.set(j,i,delta_weight);
            }
        }
       layer_weight[weight_layer] = Matrix.plus_matrix(layer_weight[weight_layer],change_weight)  ;
    }

    public void delta_weight_hiddennode(int layer){

        int weight_layer = layer-1;
        Matrix change_weight =  new Matrix(node[layer].length, node[layer-1].length ,false);
        //TODO
        int diff_fn = 1;

        //mutiply matrix
        for (int j = 0; j < error_n.get(0).length ; j++){
            for(int i=0;i< node[layer-1].length ; i++)
            {
                double delta_weight = learning_rate * (local_gradient_node[layer][j] * diff_fn * node[layer-1][i] );
                change_weight.set(j,i,delta_weight);
            }
        }
        layer_weight[weight_layer] = Matrix.plus_matrix(layer_weight[weight_layer],change_weight)  ;
    }

    public void node_eval(int layer){
        // W r_c X N r_1 = N+1 r_1
        if(   layer_weight[layer].cols != node[layer].length){
            System.out.println("invalid matrix");
            return;
        }

        double  sum_inputnode;
        Double  outputnode[] = new Double[neural_type[layer+1]];

        //mutiply matrix
        for (int j = 0; j < neural_type[layer+1] ; j++){
            double sum=0;
            for(int k=0;k<node[layer].length;k++)
            {
                //w_ji : weight from input neuron j to neron i : in each layer
                sum+=  layer_weight[layer].data[j][k]  * node[layer][k] ;
            }
            // V_j = sum all input*weight i->j + biases
            sum_inputnode = sum + biases;
            // Y_j =  nonlinear ity activation_fn associated with neuron j //  Y_j  : output each node
            outputnode[j] = actication_fn(sum_inputnode); // transpose
        }
        // O_k  =  output of neuron_node k in each layer
        node[layer+1] = outputnode;
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

//        return 1 / (1 + Math.exp(-x));\
        //TODO
        return  x;
    }
}
