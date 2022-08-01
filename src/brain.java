import java.util.ArrayList;

public class brain {
    private final int[] neural_type;
    private final ArrayList<Double[]> train_dataset;
    private final ArrayList<Double[]> train_desired_data;


    private final int maxEpoch;
    private final double minError;
    private final double learning_rate;
    private final double moment_rate;


    private final ArrayList<Double[]> error_n = new ArrayList<>(); //sum of squared error at iteration n (sse)
    private double avg_error_n = 1000000000 ; // average sse of all epoch
    private double biases; // threshold connected : biases

    private Matrix[] layer_weight  ;
    private Matrix[] change_weight;


//    private Matrix[] old_change_weight;
    private Double[][]  node  ;
    private Double[][]  suminput_of_node  ;
    private Double[][]  local_gradient_node  ;




    public brain(String _neural_type ,ArrayList<Double[]> _train_dataset,ArrayList<Double[]> _train_desired_data  , int _maxEpoch , double _minError ,double _biases, double  _learning_rate , double _moment_rate){

        String[] splitArray = _neural_type.split(",");
        int[] array = new int[splitArray.length];
        for (int i = 0; i < splitArray.length; i++) array[i] = Integer.parseInt(splitArray[i]);

        this.neural_type = array;

        init_Structor();

        this.train_dataset = _train_dataset;
        this.train_desired_data = _train_desired_data;
        this.maxEpoch = _maxEpoch;
        this.minError = _minError;
        this.biases = _biases;
        this.learning_rate = _learning_rate;
        this.moment_rate = _moment_rate;


    }
    private void init_Structor(){
        node = new Double[neural_type.length][];
        local_gradient_node = new Double[neural_type.length][];
        for (int i = 0; i < neural_type.length; i++) {
            node[i] = new Double[neural_type[i]];
            local_gradient_node[i] = new Double[neural_type[i]];
        }

        layer_weight = new Matrix[neural_type.length-1];
        change_weight = new Matrix[neural_type.length-1];
        for (int layer = 0; layer < layer_weight.length; layer++) {
            Matrix weight = new Matrix(neural_type[layer+1],neural_type[layer] ,true);
            Matrix change = new Matrix(neural_type[layer+1],neural_type[layer] ,false);
            layer_weight[layer] = weight;
            change_weight[layer] = change;
        }

    }

    public void train(){
        int N =0;
        while (N < maxEpoch && avg_error_n > minError){ //

            error_n.clear();




            for(int data = 0; data < train_dataset.size() ; data++) {
              //setup input data
                int ran_dataset_i = (int) (Math.random() * ((train_dataset.size()) ));

                //set dataset value to input node
                for(int input_i = 0 ; input_i < neural_type[0] ; input_i ++){
                    node[0][input_i] = train_dataset.get(ran_dataset_i)[input_i];
                }

                //cal ∑(input x weight) -> activation_Fn  for each neuron_node
                forward_pass();

                get_error(ran_dataset_i , N);
                backward_pass();

                double d = train_desired_data.get(ran_dataset_i)[0]*700 ;
                double g = node[node.length-1][0]*700;
                System.out.println("desired:" + d + " get: "+ g + " error_n: " + (d-g));

            }

            // avg_E(n) = 1/N ∑ E(n)  : avg (sse)
            double sum = 0.0;
            for (Double[] doubles : error_n) {
                sum += Math.pow(doubles[0], 2);
            }
            avg_error_n =  sum / (error_n.size());


            System.out.println("epoch: " + N + "   avg_error_n: " + avg_error_n);
            System.out.println("===================================================");
            N++; // next epoch
        }

        System.out.println("avg_error_n final : " + avg_error_n);

        save_weight();
    }



    private void forward_pass(){
        for(int layer = 0; layer < neural_type.length-1 ; layer++) {

            // W r_c X N r_1 = N+1 r_1
            if(   layer_weight[layer].cols != node[layer].length){
                System.out.println("invalid matrix");
                return;
            }

            double  sum_input;
            Double[] sum_inputnode = new Double[neural_type[layer+1]];

            //mutiply matrix
            for (int j = 0; j < neural_type[layer+1] ; j++){
                double sum=0;
                for(int k=0;k<node[layer].length;k++)
                {
                    //w_ji : weight from input neuron j to neron i : in each layer
                     sum+=  layer_weight[layer].data[j][k]  *  activation_fn( node[layer][k])  ;
                }
                // V_j = sum all input*weight i->j + biases
                sum_input = sum + biases;
                sum_inputnode[j] = sum_input;
            }
            // O_k  =  output of neuron_node k in each layer
            node[layer+1] = sum_inputnode;

        }
    }

    private void get_error(int ran_dataset_i , int N) {

        // ∑E(n) = 1/2 ∑ e^2   : sum of squared error at iteration n (sse)
        int number_outputn_node  =   node[node.length-1].length;
        Double[] errors = new Double[number_outputn_node];
        for ( int outnode_j = 0 ; outnode_j < number_outputn_node ; outnode_j++) {
            //train_desired_data => d_j desired output for neuron_node j at iteration N // it have "one data"
            //e_j  = error at neuron j at iteration N
            double desired = train_desired_data.get(ran_dataset_i)[0];
            errors[outnode_j] =  desired -node[node.length-1][outnode_j];
            //TODO
            double diff_fn  = 1.0;//diff_activation_fn(node.length-1 , outnode_j);
            local_gradient_node[node.length-1][outnode_j] =  errors[outnode_j] *  diff_fn;
        }
        error_n.add(errors);
    }


    private void backward_pass() {
        //⊃ sse/w_j =  ⊃ ∑(n) / ⊃ e_j  *  ⊃ e_j / ⊃ Y_j  *  ⊃ Y_j / ⊃ V_j  *  ⊃ V_j / ⊃ w_ji
        //⊃ sse/w_j =     (e_j(n))     *           -1    * diff Y_j(sum_input)  * Y_i

        // Y_j is  linear_fn
        //diff Y_j =  linear_fn -> 1
        //TODO
        // ɳ =  learning rate
        // ∆weight_ji = - ɳ (  ⊃ sse/w_j )
        // ∆weight_ji  =  ɳ [ (e_j(n)) * diff Y_j(sum_input) * Y_i ]
        // ∆weight_ji = ∆weight_ji(old) + ∆weight_ji
        // wji_next = wji_now + ∆weight_ji

        // output change_weight
        int output_layer = node.length-1;
         delta_weight_outputnode(output_layer);

        //local gradient output_k= e_k * diff Y_k    ::    local gradient hidden_j = diff Y_j * ∑ (  W_kj  * (- ?) l_g k)
        local_gradient();
        for (int layer = node.length-3 ; layer >= 0  ; layer--) {
            // hidden layer change_weight
            // ∆weight_ji =   ɳ *  local_gradient_j * Y_i
            delta_weight_hiddennode(layer);
        }

        for (int weight_layer = layer_weight.length-1 ; weight_layer >= 0  ; weight_layer--) {

                   layer_weight[weight_layer] = Matrix.plus_matrix(layer_weight[weight_layer], change_weight[weight_layer])  ;

        }
    }



    public void delta_weight_outputnode(int layer){
        //TODO it should to global?
        int weight_layer = layer-1;
//        output_change_weight =  new Matrix(error_n.get(0).length, node[layer-1].length ,false);
//
        //mutiply matrix
        for (int j = 0; j < error_n.get(0).length ; j++){

            double diff_fn  = diff_activation_fn(layer , j);
            for(int i=0;i< node[layer-1].length ; i++)
            {
                double old_weight =  moment_rate * change_weight[weight_layer].data[j][i];
                double delta_weight =  learning_rate * (error_n.get(0)[j] * diff_fn * activation_fn(node[layer-1][i])) ;
                double delta =  old_weight  +  delta_weight;
                change_weight[weight_layer].set(j,i,delta);
            }
        }

    }
    private void local_gradient() {
        for (int layer = layer_weight.length-1 ; layer >= 0  ; layer--) {
            for (int j = 0; j < node[layer].length   ; j++){


                double sum_j = 0;
                for(int k=0;k< node[layer+1].length  ; k++)
                {
                    // TODO
                    sum_j +=  ( local_gradient_node[layer+1][k])  *  layer_weight[layer].data[k][j] ;
                }

                double diff_fn  = diff_activation_fn(layer , j);
                local_gradient_node[layer][j] = sum_j * diff_fn;
            }
        }
    }
    public void delta_weight_hiddennode(int weight_layer){


        int node_layer = weight_layer+1;

        //mutiply matrix

        for (int j = 0; j <  node[node_layer].length ; j++){
            for(int i=0;i< node[node_layer-1].length ; i++)
            {
                double old_weight =  moment_rate * change_weight[weight_layer].data[j][i];
                double delta_weight = learning_rate * ( local_gradient_node[node_layer][j] * activation_fn(node[node_layer-1][i])) ;
                double delta = old_weight  +  delta_weight;

                change_weight[weight_layer].set(j,i,delta);
            }
        }

    }



    private void save_weight() {
        //TODO
//        good_weight = layer_weight;
    }

    public void test(ArrayList<Double[]> _test_dataset,ArrayList<Double[]> _test_desired_data){

        //setup input data
        for(int test_i = 0; test_i < _test_dataset.size()-1 ; test_i++) {

            //set dataset value to input node
            for (int input_i = 0; input_i < neural_type[0]; input_i++) {
                node[0][input_i] = _test_dataset.get(test_i)[input_i];
            }

            forward_pass();

             double d = _test_desired_data.get(test_i)[0]*700 ;
             double g = node[node.length-1][0]*700;
            System.out.println("desired:" + d  + "  get:" + g   + " err:" + (d-g));
        }
    }



    public double activation_fn(Double x){
        //TODO
        return Math.max(0.01,x);
//        return (Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x)); // Tanh
//        return 1 / (1 + Math.exp(-x)); //sigmoid
//         return  x; // linear
    }

    public double diff_activation_fn(int layer , int j ){
        //TODO
        if(node[layer][j]<=0){
            return 0.01;
        }else{
            return 1;
        }

//        return   1.0 - (node[layer][j] * node[layer][j] ); //Tanh
//      return node[layer][j] * (1 - node[layer][j]); //sigmoid
//      return 1; //linear
    }

}
