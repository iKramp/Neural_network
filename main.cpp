#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

#define CONST_E 2.7182818284590452353602874713527

//network with nodes 784-32-16-10
//  and with weights 25088-512-160
struct node{
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

    weights[0].resize(25088);
    weights[1].resize(512);
    weights[2].resize(160);
    for(int j = 0; j < 3; j++)
        for(float & i : weights[j])
            i = (rand() % 2) * 2 - 1;
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
            all_images[i].pixels[j] = (float)images_data[i * 784 + j + 16] / 255;
        }
    }
    delete []images_data;
    delete []label_data;

    /*auto rng = std::default_random_engine {};
    std::shuffle(std::begin(all_images), std::end(all_images), rng);*/
}

float sigmoid_func(float input){
    return 1 / (1 + pow(CONST_E, -input));
}

float sigmoid_derivative(float input){
    return sigmoid_func(input) * (1 - sigmoid_func(input));
}


void guess_image(images& image, std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    for(int i = 0; i < 784; i++){
        neurons[0][i].value = image.pixels[i];
    }
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < neurons[i + 0].size() * neurons[i + 1].size(); j++){
            neurons[i + 1][j / neurons[i].size()].value += neurons[i][j %  neurons[i].size()].value * weights[i][j];
            if((j + 1) %  neurons[i].size() == 0){
                neurons[i + 1][j / neurons[i].size()].value += neurons[i + 1][j / neurons[i].size()].bias;
                neurons[i + 1][j / neurons[i].size()].value = sigmoid_func(neurons[i + 1][j / neurons[i].size()].value);
            }
        }
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
    for(auto & neuron_layer : neurons){
        for(auto & neuron : neuron_layer){
            float test = neuron.bias;
            char* data_2 = new char [4];
            data_2[0] = *(char*)&neuron.bias;
            std::string test_string = data_2;
            *(float*)&data[index] = neuron.bias;
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
}

void backprop_layer(int layer, float cost, std::vector<std::vector<node>>& neurons, std::vector<float>& weights, std::vector<std::vector<node>>& new_neurons, std::vector<float>& new_weights){
    for(int i = 0; i < weights.size(); i++){//setting up weights
        float dc_dw = neurons[layer][i % neurons[layer].size()].value;
        dc_dw *= sigmoid_derivative(weights[i] * neurons[layer][i % neurons[layer].size()].value + neurons[layer + 1][i / neurons[layer].size()].bias);
        dc_dw *= (2 * neurons[layer + 1][i / neurons[layer].size()].value - new_neurons[layer + 1][i / neurons[layer].size()].value);
        float dw = -cost / dc_dw;
        new_weights[i] += weights[i] + dw;
    }
    for(int i = 0; i < weights.size(); i++){//setting up biases
        float dc_db = 1;
        dc_db *= sigmoid_derivative(weights[i] * neurons[layer][i % neurons[layer].size()].value + neurons[layer + 1][i / neurons[layer].size()].bias);
        dc_db *= (2 * neurons[layer + 1][i / neurons[layer].size()].value - new_neurons[layer + 1][i / neurons[layer].size()].value);
        float db = -cost / dc_db;
        new_neurons[layer + 1][1 / neurons[layer].size()].bias += neurons[layer + 1][1 / neurons[layer].size()].bias + db;
    }
    for(int i = 0; i < weights.size(); i++){//setting up next layer
        float dc_dv = weights[i];
        dc_dv *= sigmoid_derivative(weights[i] * neurons[layer][i % neurons[layer].size()].value + neurons[layer + 1][i / neurons[layer].size()].bias);
        dc_dv *= (2 * neurons[layer + 1][i / neurons[layer].size()].value - new_neurons[layer + 1][i / neurons[layer].size()].value);
        float dv = -cost / dc_dv;
        new_neurons[layer][1 % neurons[layer].size()].value += neurons[layer + 1][1 % neurons[layer].size()].bias + dv;
    }
}


void backprop(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, std::vector<images>& all_images, int start_images){
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


    for(int i = 0 + start_images; i < 100 + start_images; i++) {
        float cost = 0;
        guess_image(all_images[i], neurons, weights);
        for (int i = 0; i < 10; i++) {
            new_neurons[3][i].value = i == all_images[i].label ? 1 : 0;//value is used to store what the values should be
            cost += pow(new_neurons[3][i].value - neurons[3][i].value, 2);
        }

        for (int i = 2; i >= 0; i--) {
            backprop_layer(i, cost, neurons, weights[i], new_neurons, new_weights[i]);
        }
    }
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons.size(); j++) {
            float test = new_neurons[i][j].bias;
            neurons[i][j].bias += new_neurons[i][j].bias / 100;
        }
    }
    for(int i = 0; i < weights.size(); i++) {
        for(int j = 0; j < weights.size(); j++) {
            float test = new_weights[i][j] / 100;
            weights[i][j] += new_weights[i][j] / 100;
        }
    }
}


int main(int argc, char *argv[]) {
    srand(time(0));
    int image_to_guess = 13;
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
            loadImages(images_path + "/testing/test_images.data", images_path + "/testing/test_labels.data", all_images);//not loading values correctly i think
            test_ai(all_images, neurons, weights);
        }else {
            loadImages(images_path + "/training/train_images.data", images_path + "/training/train_labels.data",all_images);
            for(int i = 0; i < 10; i++){
                backprop(neurons, weights, all_images, i * 10);
                std::cout << i << std::endl;
            }

        }
        if(strcmp(argv[2], "true") == 0){
            saveWeights_biases(neurons, weights, images_path + "/neural_data.wbdata");
        }


    }

    return 0;
}
