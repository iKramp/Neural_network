#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

#define CONST_E 2.7182818284590452353602874713527
#define LEARNING_CURVE 3000

//network with nodes 784-32-16-10
//  and with weights 25088-512-160
struct node{
    float half_calc_val = 0;
    float value = 0.5;
    float bias = 0;
};

struct images{
    float pixels[784];
    unsigned char label;
};



void initialize_array(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    neurons[0].resize(784);
    neurons[1].resize(32);
    neurons[2].resize(16);
    neurons[3].resize(10);
    for(int j = 1; j < 4; j++)
        for(auto & i : neurons[j])
            i.bias = (float)(rand() % 64) / 32;

    weights[0].resize(25088);
    weights[1].resize(512);
    weights[2].resize(160);
    for(int j = 0; j < 3; j++)
        for(auto & i : weights[j])
            i = ((float)(rand() % 64) - 31.5f) / 64;
}

void loadImages(const std::string& image_path, const std::string& label_path, std::vector<images>& all_images){
    std::ifstream images_file, label_file;
    char* images_data;
    char* label_data;

    images_file.open(image_path, std::ios::in);
    label_file.open(label_path, std::ios::in);
    if(!images_file.is_open() || !label_file.is_open())
        throw;//don't have time to throw a meaningful exception so i just throw lol

    images_file.seekg(0, std::ios::end);
    long size = (int)images_file.tellg();
    images_data = new char[size];
    images_file.seekg(0, std::ios::beg);
    images_file.read(images_data, size);

    label_file.seekg(0, std::ios::end);
    size = (int)label_file.tellg();
    label_data = new char[size];
    label_file.seekg(0, std::ios::beg);
    label_file.read(label_data, size);

    images_file.close();
    label_file.close();

    size = ((unsigned char)images_data[6] << 8) + (unsigned char)images_data[7];


    for(int i = 0; i < size; i++){
        all_images.resize(size);
        all_images[i].label = label_data[i + 8];
        for(int j = 0; j < 784; j++){
            all_images[i].pixels[j] = ((float)(*(unsigned char*)(char*)&images_data[i * 784 + j + 16]) + 1) / 256;
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(all_images.begin(), all_images.end(), g);

    delete []images_data;
    delete []label_data;

}

float ReLU_func(float input){
    return std::max(input, input / 64);
}

float ReLU_derivative(float input){
    return input < 0 ? 0.0156 : 1;

}

float sigmoid_func(float input){
    return 0.98 / (1 + std::exp(-input)) + 0.01;
}

float sigmoid_derivative(float input){
    return sigmoid_func(input) * (1 - sigmoid_func(input));
}

float sigmoid_cost(std::vector<node>& last_layer, int label){//doesn't work
    float sum = 0;
    for(int i = 0; i < last_layer.size(); i++){
        sum += std::pow(label == i ? 1 : 0 - last_layer[i].value, 2);
    }
    return sum;
}

void guess_image(images& image, std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    for(int i = 0; i < 784; i++){
        neurons[0][i].value = image.pixels[i];
    }
    for(int i = 1; i < 4; i++){
        for(int j = 0; j < neurons[i].size(); j++){
            neurons[i][j].half_calc_val = 0;
        }
    }
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < neurons[i].size() * neurons[i + 1].size(); j++){
            neurons[i + 1][j / neurons[i].size()].half_calc_val += neurons[i][j %  neurons[i].size()].value * weights[i][j];
            if((j + 1) %  neurons[i].size() == 0){
                neurons[i + 1][j / neurons[i].size()].half_calc_val += neurons[i + 1][j / neurons[i].size()].bias;
                neurons[i + 1][j / neurons[i].size()].value = ReLU_func(neurons[i + 1][j / neurons[i].size()].half_calc_val);
            }
        }
    }
    for(int j = 0; j < neurons[2].size() * neurons[3].size(); j++){
        neurons[3][j / neurons[2].size()].half_calc_val += neurons[2][j %  neurons[2].size()].value * weights[2][j];
        if((j + 1) %  neurons[2].size() == 0)
            neurons[3][j / neurons[2].size()].half_calc_val += neurons[3][j / neurons[2].size()].bias;
    }
    for(auto& neuron : neurons[3]){
        neuron.value = sigmoid_func(neuron.half_calc_val);
    }

}

void loadWeights_biases(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, const std::string& path){
    char* data = new char[106408];

    std::ifstream wbfile;
    wbfile.open(path, std::ios::in);
    wbfile.read(data, 106408);
    wbfile.close();


    int index = 0;
    for(auto & neuron_layer : neurons){
        for(auto & neuron : neuron_layer){
            neuron.bias = *(float*)&data[index];
            index += 4;
        }
    }
    for(auto & weight_layer : weights){
        for(auto & weight : weight_layer){
            weight = *(float*)&data[index];
            index += 4;
        }
    }

    delete[] data;
}

void saveWeights_biases(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, const std::string& path){

    char* data = new char[106408];
    int index = 0;
    for(int i = 0; i < neurons.size(); i++){
        for(int j = 0; j < neurons[i].size(); j++){
            *(float*)&data[index] = neurons[i][j].bias;
            index += 4;
        }
    }
    for(auto & weight_layer : weights){
        for(auto & weight : weight_layer){
            *(float*)&data[index] = weight;
            index += 4;
        }
    }

    std::ofstream wbfile;
    wbfile.open(path, std::ios::out | std::ios::trunc);
    wbfile.write(data, 106408);
    wbfile.close();

    delete[] data;
}

void test_ai(std::vector<images>& all_images, std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    int image_number = 0;
    int correct_answers = 0;
    float average_cost = 0;
    for(auto& image : all_images){
        image_number++;
        guess_image(image, neurons, weights);
        int best_guess_number;
        float best_guess_value = 0;

        for(int i = 0; i < 10; i++){
            if(neurons[3][i].value > best_guess_value){
                best_guess_value = neurons[3][i].value;
                best_guess_number = i;
            }
        }

        average_cost += sigmoid_cost(neurons[3], image.label);

        if(best_guess_number == image.label)
            correct_answers++;
        float correct_percentage = 0;
        if(correct_answers)
            correct_percentage = (float)correct_answers / (float)image_number;


        if(image_number % 100 == 0) {
            std::cout << image_number << std::endl;
        }
        if(image_number % 9999 == 0) {
            std::cout << std::to_string(correct_percentage).substr(0, 5);
        }
    }
    average_cost /= all_images.size();
    std::cout << std::endl << average_cost;
}

void backprop_layer_ReLU(int layer, std::vector<std::vector<node>>& neurons, std::vector<float>& weights, std::vector<std::vector<node>>& new_neurons, std::vector<float>& new_weights){
    for(int i = 0; i < neurons[layer + 1].size(); i++){//setting up biases
        float dc_db = ReLU_derivative(neurons[layer + 1][i].half_calc_val);
        dc_db *= 2 * (neurons[layer + 1][i].value - new_neurons[layer + 1][i].value);
        new_neurons[layer + 1][i].bias += dc_db;
    }

    for(int last = 0; last < neurons[layer + 1].size(); last++){
        float common_dc_dx = ReLU_derivative(neurons[layer + 1][last].half_calc_val);
        common_dc_dx *= (2 * (neurons[layer + 1][last].value - new_neurons[layer + 1][last].value));
        for(int prev = 0; prev < neurons[layer].size(); prev++){
            int i = last * neurons[layer].size() + prev;
            float dc_dw = neurons[layer][prev].value * common_dc_dx;
            float dc_dv = weights[i] * common_dc_dx;
            new_weights[i] += dc_dw;
            new_neurons[layer][prev].value += dc_dv;
            if(last == neurons[layer + 1].size() - 1)
                new_neurons[layer][prev].value += neurons[layer][prev].value;
        }
    }

    /*for(int i = 0; i < weights.size(); i++){//make it 16x faster by switching sides neurons cycle through
        float common_dc_dx = ReLU_derivative(neurons[layer + 1][i / neurons[layer].size()].half_calc_val);
        common_dc_dx *= (2 * (neurons[layer + 1][i / neurons[layer].size()].value - new_neurons[layer + 1][i / neurons[layer].size()].value));
        float dc_dw = neurons[layer][i % neurons[layer].size()].value * common_dc_dx;
        float dc_dv = weights[i] * common_dc_dx;
        new_weights[i] += dc_dw;
        new_neurons[layer][i % neurons[layer].size()].value += dc_dv;
        if(i / neurons[layer].size() == neurons[layer].size() - 1)
            new_neurons[layer][i % neurons[layer].size()].value += neurons[layer][i % neurons[layer].size()].value;
    }*/
}

void backprop_layer_sigmoid(std::vector<std::vector<node>>& neurons, std::vector<float>& weights, std::vector<std::vector<node>>& new_neurons, std::vector<float>& new_weights){
    for(int i = 0; i < neurons[3].size(); i++){//setting up biases
        float dc_db = sigmoid_derivative(neurons[3][i].half_calc_val);
        dc_db *= 2 * (neurons[3][i].value - new_neurons[3][i].value);
        new_neurons[3][i].bias += dc_db;
    }
    for(int last = 0; last < neurons[3].size(); last++){
        float common_dc_dx = sigmoid_derivative(neurons[3][last].half_calc_val);
        common_dc_dx *= (2 * (neurons[3][last].value - new_neurons[3][last].value));
        for(int prev = 0; prev < neurons[2].size(); prev++){
            int i = last * neurons[2].size() + prev;
            float dc_dw = neurons[2][prev].value * common_dc_dx;
            float dc_dv = weights[i] * common_dc_dx;
            new_weights[i] += dc_dw;
            new_neurons[2][prev].value += dc_dv;
            if(last == neurons[3].size() - 1)
                new_neurons[2][prev].value += neurons[2][prev].value;
        }
    }

    /*for(int i = 0; i < weights.size(); i++){//setting up next layer
        float common_dc_dx = sigmoid_derivative(neurons[3][i / neurons[2].size()].half_calc_val);
        common_dc_dx *= (2 * (neurons[3][i / neurons[2].size()].value - new_neurons[3][i / neurons[2].size()].value));
        float dc_dw = neurons[2][i % neurons[2].size()].value * common_dc_dx;
        float dc_dv = weights[i] * common_dc_dx;
        new_weights[i] += dc_dw;
        new_neurons[2][i % neurons[2].size()].value += dc_dv;
        if(i / neurons[2].size() == neurons[2].size() - 1)
            new_neurons[2][i % neurons[2].size()].value += neurons[2][i % neurons[2].size()].value;
    }*/
}

void backprop(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, std::vector<images>& all_images, int start_images, float& average_cost){
    std::vector<std::vector<node>> new_neurons(neurons.size());
    std::vector<std::vector<float>> new_weights(weights.size());
    for(int i = 0; i < neurons.size(); i++) {
        new_neurons[i].resize(neurons[i].size());
        for(auto & j : new_neurons[i]) {
            j.value = 0;
            j.bias = 0;
        }
    }
    for(int i = 0; i < weights.size(); i++) {
        new_weights[i].resize(weights[i].size());
        for(auto & j : new_weights[i]) {
            j = 0;
        }
    }
    float cost = 0;

    for(int i = 0 + start_images; i < 100 + start_images; i++) {

        guess_image(all_images[i], neurons, weights);

        for(int j = 0; j < 2; j++)
            for(int k = 0; k < new_neurons[j].size(); k++)
                new_neurons[j][k].value = 0;

        for (int j = 0; j < 10; j++) {
            new_neurons[3][j].value = j == all_images[i].label ? 1 : 0;
        }
        cost += sigmoid_cost(neurons[3], all_images[i].label);

        backprop_layer_sigmoid(neurons, weights[2], new_neurons, new_weights[2]);
        for (int j = 1; j >= 0; j--) {
            backprop_layer_ReLU(j, neurons, weights[j], new_neurons, new_weights[j]);
        }
    }
    cost /= 100;
    average_cost += cost;
    for(int i = 1; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            neurons[i][j].bias += ((- cost / new_neurons[i][j].bias) / LEARNING_CURVE) / 100;
        }
    }
    for(int i = 0; i < weights.size(); i++) {
        for(int j = 0; j < weights[i].size(); j++) {
            weights[i][j] += ((-cost / new_weights[i][j]) / LEARNING_CURVE) / 100;
        }
    }
}


int main(int argc, char *argv[]) {//current accuracy of 0,0891%, average cost of 600 is:
    srand(time(0));// delta C of:                                              .
    std::vector<std::vector<node>> neurons(4);
    std::vector<std::vector<float>> weights(3);
    std::vector<images> all_images;
    std::string images_path = "/home/nejc/CLionProjects/Neural_network/AI_images_set";

    initialize_array(neurons, weights);

    if(argc != 4)
        throw;//don't have time to throw a meaningful exception so i just throw lol
    else{
        if(strcmp(argv[1], "true") == 0){
            loadWeights_biases(neurons, weights, images_path + "/neural_data.wbdata");
        }
        if(strcmp(argv[3], "test") == 0){
            loadImages(images_path + "/testing/test_images.data", images_path + "/testing/test_labels.data", all_images);
            test_ai(all_images, neurons, weights);
        }else {
            loadImages(images_path + "/training/train_images.data", images_path + "/training/train_labels.data",all_images);
            float temp_cost = 3.2;
            for(int j = 0; j < 20; j++) {
                float average_cost = 0;
                for (int i = 0; i < 600; i++) {
                    backprop(neurons, weights, all_images, i * 100, average_cost);
                    std::cout << j << " " << i << std::endl;
                }
                average_cost /= 600;
                std::cout << "average cost:" << average_cost << std::endl;
                if(average_cost < temp_cost) {
                    saveWeights_biases(neurons, weights, images_path + "/neural_data.wbdata");
                    temp_cost = average_cost;
                }
            }

        }
        if(strcmp(argv[2], "true") == 0){
            saveWeights_biases(neurons, weights, images_path + "/neural_data.wbdata");
        }


    }

    return 0;
}
