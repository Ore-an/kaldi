// nnet3/nnet-chain-diagnostics.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet3/nnet-chain-diagnostics.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetChainComputeProb::NnetChainComputeProb(
    const NnetComputeProbOptions &nnet_config,
    const chain::ChainTrainingOptions &chain_config,
    const std::vector<fst::StdVectorFst> &den_fst,
    const std::vector<std::string> &den_to_output,
    const Nnet &nnet):
    nnet_config_(nnet_config),
    chain_config_(chain_config),
    nnet_(nnet),
    compiler_(nnet, nnet_config_.optimize_config, nnet_config_.compiler_config),
    deriv_nnet_owned_(true),
    deriv_nnet_(NULL),
    num_minibatches_processed_(0) {
  KALDI_ASSERT(den_fst.size() == den_to_output.size());
  // Initialize den_graph using num_pdf in corresponding output node in network.
  for (int32 fst_ind = 0; fst_ind < den_fst.size(); fst_ind++) {
    chain::DenominatorGraph den_graph(den_fst[fst_ind],
      nnet_.OutputDim(den_to_output[fst_ind]));
    den_graph_.insert(std::make_pair(den_to_output[fst_ind], den_graph));
  }
  if (nnet_config_.compute_deriv) {
    deriv_nnet_ = new Nnet(nnet_);
    ScaleNnet(0.0, deriv_nnet_);
    SetNnetAsGradient(deriv_nnet_); // force simple update
  } else if (nnet_config_.store_component_stats) {
    KALDI_ERR << "If you set store_component_stats == true and "
	      << "compute_deriv == false, use the other constructor.";
  }
}

NnetChainComputeProb::NnetChainComputeProb(
    const NnetComputeProbOptions &nnet_config,
    const chain::ChainTrainingOptions &chain_config,
    const std::vector<fst::StdVectorFst> &den_fst,
    const std::vector<std::string> &den_to_output,
    Nnet *nnet):
    nnet_config_(nnet_config),
    chain_config_(chain_config),
     nnet_(*nnet),
    compiler_(*nnet, nnet_config_.optimize_config, nnet_config_.compiler_config),
    deriv_nnet_owned_(false),
    deriv_nnet_(nnet),
    num_minibatches_processed_(0) {
  KALDI_ASSERT(den_fst.size() == den_to_output.size());
  // Initialize den_graph using num_pdf in corresponding output node in network.
  for (int32 fst_ind = 0; fst_ind < den_fst.size(); fst_ind++) {
    chain::DenominatorGraph den_graph(den_fst[fst_ind],
      nnet_.OutputDim(den_to_output[fst_ind]));
    den_graph_.insert(std::make_pair(den_to_output[fst_ind], den_graph));
  }
    KALDI_ASSERT(nnet_config.store_component_stats && !nnet_config.compute_deriv);
  }
  
const Nnet &NnetChainComputeProb::GetDeriv() const {
  if (!nnet_config_.compute_deriv)
    KALDI_ERR << "GetDeriv() called when no derivatives were requested.";
  return *deriv_nnet_;
}

NnetChainComputeProb::~NnetChainComputeProb() {
  if (deriv_nnet_owned_)
  delete deriv_nnet_;  // delete does nothing if pointer is NULL.
}

void NnetChainComputeProb::Reset() {
  num_minibatches_processed_ = 0;
  objf_info_.clear();
  if (deriv_nnet_) {
    ScaleNnet(0.0, deriv_nnet_);
    SetNnetAsGradient(deriv_nnet_);
  }
}

void NnetChainComputeProb::Compute(const NnetChainExample &chain_eg) {
  bool need_model_derivative = nnet_config_.compute_deriv,
      store_component_stats = nnet_config_.store_component_stats;
  ComputationRequest request;
  // if the options specify cross-entropy regularization, we'll be computing
  // this objective (not interpolated with the regular objective-- we give it a
  // separate name), but currently we won't make it contribute to the
  // derivative-- we just compute the derivative of the regular output.
  // This is because in the place where we use the derivative (the
  // model-combination code) we decided to keep it simple and just use the
  // regular objective.
  bool use_xent_regularization = (chain_config_.xent_regularize != 0.0),
      use_xent_derivative = false;
  GetChainComputationRequest(nnet_, chain_eg, need_model_derivative,
                             store_component_stats, use_xent_regularization,
                             use_xent_derivative, &request);
  const NnetComputation *computation = compiler_.Compile(request);
  NnetComputer computer(nnet_config_.compute_config, *computation,
                        nnet_, deriv_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet_, chain_eg.inputs);
  computer.Run();
  this->ProcessOutputs(chain_eg, &computer);
  if (nnet_config_.compute_deriv)
    computer.Run();
}

void NnetChainComputeProb::ProcessOutputs(const NnetChainExample &eg,
                                         NnetComputer *computer) {
  // There will normally be just one output here, named 'output',
  // but the code is more general than this.
  std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetChainSupervision &sup = *iter;
    int32 node_index = nnet_.GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_.IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);
    bool use_xent = (chain_config_.xent_regularize != 0.0);
    std::string xent_name = sup.name + "-xent";  // typically "output-xent".
    CuMatrix<BaseFloat> nnet_output_deriv, xent_deriv;
    if (nnet_config_.compute_deriv)
      nnet_output_deriv.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                               kUndefined);
    if (use_xent)
      xent_deriv.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                        kUndefined);

    BaseFloat tot_like, tot_l2_term, tot_weight;

    ComputeChainObjfAndDeriv(chain_config_, den_graph_[sup.name],
                             sup.supervision, nnet_output,
                             &tot_like, &tot_l2_term, &tot_weight,
                             (nnet_config_.compute_deriv ? &nnet_output_deriv :
                              NULL), (use_xent ? &xent_deriv : NULL));

    // note: in this context we don't want to apply 'sup.deriv_weights' because
    // this code is used only in combination, where it's part of an L-BFGS
    // optimization algorithm, and in that case if there is a mismatch between
    // the computed objective function and the derivatives, it may cause errors
    // in the optimization procedure such as early termination.  (line search
    // and conjugate gradient descent both rely on the derivatives being
    // accurate, and don't fail gracefully if the derivatives are not accurate).

    ChainObjectiveInfo &totals = objf_info_[sup.name];
    totals.tot_weight += tot_weight;
    totals.tot_like += tot_like;
    totals.tot_l2_term += tot_l2_term;

    if (nnet_config_.compute_deriv)
      computer->AcceptInput(sup.name, &nnet_output_deriv);

    if (use_xent) {
      ChainObjectiveInfo &xent_totals = objf_info_[xent_name];
      // this block computes the cross-entropy objective.
      const CuMatrixBase<BaseFloat> &xent_output = computer->GetOutput(
          xent_name);
      // at this point, xent_deriv is posteriors derived from the numerator
      // computation.  note, xent_deriv has a factor of '.supervision.weight',
      // but so does tot_weight.
      BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);
      xent_totals.tot_weight += tot_weight;
      xent_totals.tot_like += xent_objf;
    }
    num_minibatches_processed_++;
  }
}

bool NnetChainComputeProb::PrintTotalStats() const {
  bool ans = false;
  unordered_map<std::string, ChainObjectiveInfo, StringHasher>::const_iterator
      iter, end;
  iter = objf_info_.begin();
  end = objf_info_.end();
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    int32 node_index = nnet_.GetNodeIndex(name);
    KALDI_ASSERT(node_index >= 0);
    const ChainObjectiveInfo &info = iter->second;
    BaseFloat like = (info.tot_like / info.tot_weight),
        l2_term = (info.tot_l2_term / info.tot_weight),
        tot_objf = like + l2_term;
    if (info.tot_l2_term == 0.0) {
      KALDI_LOG << "Overall log-probability for '"
                << name << "' is "
                << like << " per frame"
                << ", over " << info.tot_weight << " frames.";
    } else {
      KALDI_LOG << "Overall log-probability for '"
                << name << "' is "
                << like << " + " << l2_term << " = " << tot_objf << " per frame"
                << ", over " << info.tot_weight << " frames.";
    }
    if (info.tot_weight > 0)
      ans = true;
  }
  return ans;
}


const ChainObjectiveInfo* NnetChainComputeProb::GetObjective(
    const std::string &output_name) const {
  unordered_map<std::string, ChainObjectiveInfo, StringHasher>::const_iterator
      iter = objf_info_.find(output_name);
  if (iter != objf_info_.end())
    return &(iter->second);
  else
    return NULL;
}

double NnetChainComputeProb::GetTotalObjective(double *tot_weights) const {
  double tot_objectives = 0.0;
  unordered_map<std::string, ChainObjectiveInfo, StringHasher>::const_iterator
    iter = objf_info_.begin(),
    end = objf_info_.end();
  for (; iter != end; ++iter) {
    tot_objectives += iter->second.tot_like + iter->second.tot_l2_term;
    (*tot_weights) += iter->second.tot_weight;
  }
  return tot_objectives;
}

double NnetChainComputeProb::GetObjectiveOuts(double *tot_weights, const std::vector<std::string> &outputs) const {
  double tot_objectives = 0.0;
  unordered_map<std::string, ChainObjectiveInfo, StringHasher>::const_iterator
    iter = objf_info_.begin(),
    end = objf_info_.end();
  for (; iter != end; ++iter) {
    for (int32 out_ind = 0; out_ind < outputs.size(); out_ind++) {
      KALDI_LOG << "KILL ME NOW" << iter->first << objf_info_.find(outputs[out_ind])->first << outputs[out_ind] ;
      if (iter == objf_info_.find(outputs[out_ind])) {
        tot_objectives += iter->second.tot_like + iter->second.tot_l2_term;
        (*tot_weights) += iter->second.tot_weight;
      }
    }
  }
  KALDI_LOG << "tot w:" << *tot_weights << " tot ob" << tot_objectives;
  return tot_objectives;
}

  void RecomputeStats(const std::vector<NnetChainExample> &egs,
		      const chain::ChainTrainingOptions &chain_config_in,
		      const std::vector<fst::StdVectorFst> &den_fst,
		      const std::vector<std::string> &den_to_output,
		      Nnet *nnet) {
    KALDI_LOG << "Recomputing stats on nnet (affects batch-norm)";
    chain::ChainTrainingOptions chain_config(chain_config_in);
    if (nnet->GetNodeIndex("output-xent") != -1 &&
	chain_config.xent_regularize == 0) {
      // this forces it to compute the output for 'output-xent', which
      // means that we'll be computing batch-norm stats for any
      // components in that branch that have batch-norm.
      chain_config.xent_regularize = 0.1;
    }

    ZeroComponentStats(nnet);
    NnetComputeProbOptions nnet_config;
    nnet_config.store_component_stats = true;
    NnetChainComputeProb prob_computer(nnet_config, chain_config, den_fst, den_to_output, nnet);
    for (size_t i = 0; i < egs.size(); i++)
      prob_computer.Compute(egs[i]);
    prob_computer.PrintTotalStats();
    KALDI_LOG << "Done recomputing stats.";
  }
  
} // namespace nnet3
} // namespace kaldi
