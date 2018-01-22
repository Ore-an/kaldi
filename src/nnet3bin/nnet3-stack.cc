// nnet3bin/nnet3-stack.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"
#include <boost/algorithm/string/replace.hpp>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Stacks two nnet3 neural networks, linked by a given layer\n"
        "\n"
        "Usage:  nnet3-init [options] <existing-model-1> <existing-model-2> <connection-layer> <raw-nnet-out>\n"
        "e.g.:\n"
        " nnet3-stack a.raw b.raw tdnn_bn.batchnorm out.raw\n";

    bool binary_write = true;
    int32 srand_seed = 0;
    bool remove_out = true;
    std::string prefix = "base-";
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("remove-out", &remove_out, "Remove output layers of first network");
    po.Register("srand", &srand_seed, "Seed for random number generator");
    po.Register("prefix", &prefix, "Prefix to append to first nnet node names");
    po.Read(argc, argv);
    srand(srand_seed);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet1_rxfilename = po.GetArg(1),
        raw_nnet2_rxfilename = po.GetArg(2),
        connection_layer = po.GetArg(3),
        raw_nnet_wxfilename = po.GetArg(4);

    Nnet nnet1;
    Nnet nnet2;
    ReadKaldiObject(raw_nnet1_rxfilename, &nnet1);
    ReadKaldiObject(raw_nnet2_rxfilename, &nnet2);
    KALDI_LOG << "Read raw neural nets from "
                << raw_nnet1_rxfilename << " and "
                << raw_nnet2_rxfilename;


    if (remove_out) {
      std::vector<int32> nodes_to_remove;
      for (int32 n = 0; n < nnet1.NumNodes(); n++) {
        if (nnet1.IsOutputNode(n))
          nodes_to_remove.push_back(n);
      }
      const bool assert_no_out = false;
      nnet1.RemoveSomeNodes(nodes_to_remove, assert_no_out);
     }


// rename nodes and components of net1 so they don't clash with net2.
  for (const auto node_name : nnet1.GetNodeNames()) {
    if (node_name != "input")
      nnet1.SetNodeName(nnet1.GetNodeIndex(node_name), (prefix + node_name));
  }

  for (const auto component_name : nnet1.GetComponentNames()) {
    nnet1.SetComponentName(nnet1.GetComponentIndex(component_name), (prefix + component_name));
  }


  for (int32 n = 0; n < nnet2.GetComponentNames().size(); n++){
    Component *new_component = nnet2.GetComponent(n)->Copy();
    nnet1.AddComponent(nnet2.GetComponentName(n), new_component);
  }
  std::vector<std::string> nodes2;
  // Text representation of nodes
  const bool include_dim = false;
  nnet2.GetConfigLines(include_dim, &nodes2);

  nodes2.erase(nodes2.begin(), nodes2.begin() + 1);
  boost::replace_first(nodes2[0], "input=input", ("input=" + prefix + connection_layer));
  boost::replace_all(nodes2[0], "input,", prefix + connection_layer + ",");

  nnet1.AddNodesFromNnet(nodes2);

//
//  for (const auto i:nodes1)
//    std::cout << i + "\n";


  nnet1.RemoveOrphanNodes();
  nnet1.RemoveOrphanComponents();

  WriteKaldiObject(nnet1, raw_nnet_wxfilename, binary_write);

  KALDI_LOG << "Wrote raw neural net to "
            << raw_nnet_wxfilename;
  return 0;
 } catch(const std::exception &e) {
   std::cerr << e.what() << '\n';
   return -1;
 }

}


/*
Test script:

cat <<EOF | nnet3-init --binary=false - foo.raw
component name=affine1 type=NaturalGradientAffineComponent input-dim=72 output-dim=59
component name=relu1 type=RectifiedLinearComponent dim=59
component name=final_affine type=NaturalGradientAffineComponent input-dim=59 output-dim=298
component name=logsoftmax type=SoftmaxComponent dim=298
input-node name=input dim=18
component-node name=affine1_node component=affine1 input=Append(Offset(input, -4), Offset(input, -3), Offset(input, -2), Offset(input, 0))
component-node name=nonlin1 component=relu1 input=affine1_node
component-node name=final_affine component=final_affine input=nonlin1
component-node name=output_nonlin component=logsoftmax input=final_affine
output-node name=output input=output_nonlin
EOF

cat <<EOF | nnet3-init --binary=false foo.raw -  bar.raw
component name=affine2 type=NaturalGradientAffineComponent input-dim=59 output-dim=59
component name=relu2 type=RectifiedLinearComponent dim=59
component name=final_affine type=NaturalGradientAffineComponent input-dim=59 output-dim=298
component-node name=affine2 component=affine2 input=nonlin1
component-node name=relu2 component=relu2 input=affine2
component-node name=final_affine component=final_affine input=relu2
EOF

rm foo.raw bar.raw

 */
