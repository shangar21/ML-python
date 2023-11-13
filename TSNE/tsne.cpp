#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <limits>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<Eigen/Dense>
#include<vector>
#include<math.h>

float ShannonEntropy(Eigen::MatrixXd m){
  Eigen::MatrixXd mLog = m.array().log2();
  return (-(m.transpose() * mLog)).diagonal().sum();
}

Eigen::MatrixXd ConditionalProbabilities(Eigen::MatrixXd m, float sigma_i, int i){
  Eigen::MatrixXd mDiff = (-m).rowwise() + m.row(i);
  Eigen::MatrixXd mSquaredDiff = mDiff.array().pow(2);
  Eigen::MatrixXd mNorms = mSquaredDiff.rowwise().sum();
  Eigen::MatrixXd mNormsSigma = (-mNorms).array() / (2 * sigma_i);
  Eigen::MatrixXd probs = mNormsSigma.array().exp();
  float totalProbs = probs.sum() - 1;
  probs = probs.array() / totalProbs;
  probs.row(i).setOnes();
  return probs;
}

float GetSigmaUpperBound(Eigen::MatrixXd m, float perplexity, int i){
  float sigma = perplexity;
  Eigen::MatrixXd condProbs = ConditionalProbabilities(m, sigma, i);
  float ubTest = exp2(ShannonEntropy(condProbs));
  while(ubTest < perplexity){
    sigma *= 2;
    condProbs = ConditionalProbabilities(m, sigma, i);
    ubTest = exp2(ShannonEntropy(condProbs));
  }
  return sigma;
}

float GetSigma(Eigen::MatrixXd m, float perplexity, int i, float tolerance = 0.001){
  float max_sigma = GetSigmaUpperBound(m, perplexity, i);
  //std::cout << "max_sigma " << max_sigma << '\n';
  float min_sigma = 0;
  float mid_sigma = (max_sigma - min_sigma)/2;
  Eigen::MatrixXd condProbs = ConditionalProbabilities(m, mid_sigma, i);
  float pTest = exp2(ShannonEntropy(condProbs));
  while (abs(pTest - perplexity) > tolerance){
    if (pTest < perplexity){
      min_sigma = mid_sigma;
    }else{
      max_sigma = mid_sigma;
    }
    mid_sigma = min_sigma > 1 ? (max_sigma - min_sigma)/2 : (max_sigma + min_sigma)/2;
    condProbs = ConditionalProbabilities(m, mid_sigma, i);
    pTest = exp2(ShannonEntropy(condProbs));
  }
  return mid_sigma;
}

std::vector<float> FindAllSigmas(Eigen::MatrixXd m, float perplexity){
  std::vector<float> sigmas;
  for(int i = 0; i < m.rows(); i++){
    sigmas.push_back(GetSigma(m, perplexity, i));
  }
  return sigmas;
}

//Eigen::MatrixXd GetPairwiseAffinities(Eigen::MatrixXd m, float perplexity, int i){
//  std::vector<float> sigmas = FindAllSigmas(m, perplexity);
//  Eigen::MatrixXd pairwiseAffinities = ConditionalProbabilities(m, sigmas[i], i);
//  for(int j = 0; j < pairwiseAffinities.rows(); j++){
//    if (j != i){
//      Eigen::MatrixXd flippedAffinities = ConditionalProbabilities(m, sigmas[j], i);
//      pairwiseAffinities.row(j) = (flippedAffinities.row(i) + pairwiseAffinities.row(j))/2;
//    }
//  }
//  return pairwiseAffinities;
//}
//
//Eigen::MatrixXd GetLowDimAffinities(Eigen::MatrixXd m, int i){
//  Eigen::MatrixXd mDiff = (-m).rowwise() + m.row(i);
//  Eigen::MatrixXd mSquaredDiff = mDiff.array().pow(2);
//  Eigen::MatrixXd mNorms = mSquaredDiff.rowwise().sum();
//  Eigen::MatrixXd onePlusMNorms = mNorms.rowwise() + 1;
//  Eigen::MatrixXd probs = 1 / onePlusMNorms.rowwise();
//  float totalProbs = probs.sum() - 1;
//  probs = probs.array()/totalProbs;
//  probs.row(i).setOnes();
//  return probs;
//}
//
//float computeTSNEGrad(Eigen::MatrixXd highDimAffinities, Eigen::MatrixXd lowDimAffinities, Eigen::MatrixXd lowDimRep, int i){
//  Eigen::MatrixXd affinitiesDiff = highDimAffinities.array() - lowDimAffinities.array();
//  Eigen::MatrixXd lowDimPointDiff = lowDimRep.array() - lowDimRep.row(i);
//  Eigen::MatrixXd lowDimAffinitiesDiff = 1 / (lowDimPointDiff.array.pow(2).rowwise().sum().rowwise() + 1).rowwise();
//  Eigen::MatrixXd grad = 4 * (affinitiesDiff.array() * lowDimPointDiff.array() * lowDimAffinitiesDiff.array()).sum();
//  return grad;
//}
//
//Eigen::MatrixXd TSNE(Eigen::MatrixXd m, float perplexity, int nDimensions = 2, float lr = 0.01, float momentum = 0.9, int max_its = 10000){
//  Eigen::MatrixXd embedding = Eigen::MatrixXd::Random(m.rows(), nDimensions);
//  for(int i = 0; i < m.rows(); i++){
//    Eigen::MatrixXd pairwiseAffinities = GetPairwiseAffinities(m, perplexity, i);
//    Eigen::MatrixXd sampleY = embedding.row(i);
//    Eigen::MatrixXd sampleY_new = sampleY;
//    Eigen::MatrixXd lowDimAffinities = GetLowDimAffinities(sampleY, i);
//    float grad = computeTSNEGrad(pairwiseAffinities, lowDimAffinities, sampleY, i);
//    int count = 0;
//    while (grad != 0 && count < max_its){
//      sampleY = sampleY_new;
//      sampleY_new += lr * grad + momentum * (sampleY_new - sampleY);
//    }
//    embedding.row(i) = sampleY_new;
//  }
//  return embedding;
//}

int main(){
  Eigen::MatrixXd m = Eigen::MatrixXd::Random(100,5);
  Eigen::MatrixXd probs = ConditionalProbabilities(m, 0.5, 1);
  float shannon = ShannonEntropy(probs);
  std::cout << shannon << '\n';
  float sigma_1 = GetSigma(m, 50, 0);
  std::cout << sigma_1 << '\n';
  std::vector<float> sigmas = FindAllSigmas(m, 50);
  for (auto i : sigmas){std::cout << i << ' ';}
  std::cout << '\n';
 // Eigen::MatrixXd embedding = TSNE(m, 50);
  return 0;
}

