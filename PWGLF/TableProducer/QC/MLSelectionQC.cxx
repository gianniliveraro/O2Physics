// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//  *+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*
//  ML selection QC task
//  *+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*
//
//  This task applies ML selection over a set of V0/Casc MC indices and
//  creates QA plots.
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    gianni.shigeru.setoue.liveraro@cern.ch
//    romain.schotter@cern.ch
//    david.dobrigkeit.chinellato@cern.ch
//

#include <Math/Vector4D.h>
#include <cmath>
#include <array>
#include <cstdlib>

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/ASoA.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessPIDTables.h"
#include "PWGLF/DataModel/LFStrangenessMLTables.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "CCDB/BasicCCDBManager.h"
#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include "Tools/ML/MlResponse.h"
#include "Tools/ML/model.h"

using namespace o2;
using namespace o2::analysis;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::ml;
using std::array;
using std::cout;
using std::endl;

// For original data loops
using V0OriginalDatas = soa::Join<aod::V0Indices, aod::V0Cores, aod::V0LambdaMLScores, aod::V0GammaMLScores>;

// For derived data analysis
using V0DerivedDatas = soa::Join<aod::V0Cores, aod::V0CollRefs, aod::V0Extras, aod::V0MCDatas, aod::V0LambdaMLScores, aod::V0GammaMLScores>;

struct MLSelectionQC {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  // Axis
  // base properties
  ConfigurableAxis centralityAxis{"centralityAxis", {100, 0.0f, 100.0f}, ""};
  ConfigurableAxis BDTScoreAxis{"BDTScoreAxis", {100, 0.0f, 1.0f}, ""};
  ConfigurableAxis MCTruthAxis{"MCTruthAxis", {2, 0.0f, 2.0f}, ""};
  ConfigurableAxis axisLambdaMass{"axisLambdaMass", {200, 1.101f, 1.131f}, ""};
  ConfigurableAxis axisGammaMass{"axisGammaMass", {100, 0.0f, 0.5f}, ""};
  ConfigurableAxis axisK0ShortMass{"axisK0ShortMass", {200, 0.4f, 0.6f}, ""};
  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "p_{T} (GeV/c)"};
  
  void init(InitContext const&)
  {
    // Histograms
    histos.add("hEventCentrality", "hEventCentrality", kTH1F, {centralityAxis});

    // ML Output (Real data)
    histos.add("hLambdaBDTScore", "hLambdaBDTScore", kTH1F, {BDTScoreAxis});
    histos.add("hAntiLambdaBDTScore", "hAntiLambdaBDTScore", kTH1F, {BDTScoreAxis});
    histos.add("hGammaBDTScore", "hGammaBDTScore", kTH1F, {BDTScoreAxis});
    histos.add("hK0ShortBDTScore", "hK0ShortBDTScore", kTH1F, {BDTScoreAxis});

    // ML Output (MC data)
    histos.add("hMCTrueLambdaBDTScore", "hMCTrueLambdaBDTScore", kTH1F, {BDTScoreAxis});
    histos.add("hMCTrueAntiLambdaBDTScore", "hMCTrueAntiLambdaBDTScore", kTH1F, {BDTScoreAxis});
    histos.add("hMCTrueGammaBDTScore", "hMCTrueGammaBDTScore", kTH1F, {BDTScoreAxis});
    histos.add("hMCTrueK0ShortBDTScore", "hMCTrueK0ShortBDTScore", kTH1F, {BDTScoreAxis});
  
    // Invariant Mass
    histos.add("h2dLambdaMassVsBDTScore", "h2dLambdaMassVsBDTScore", kTH2F, {BDTScoreAxis, axisLambdaMass});
    histos.add("h2dAntiLambdaMassVsBDTScore", "h2dAntiLambdaMassVsBDTScore", kTH2F, {BDTScoreAxis, axisLambdaMass});
    histos.add("h2dGammaMassVsBDTScore", "h2dGammaMassVsBDTScore", kTH2F, {BDTScoreAxis, axisGammaMass});
    histos.add("h2dK0ShortMassVsBDTScore", "h2dK0ShortMassVsBDTScore", kTH2F, {BDTScoreAxis, axisK0ShortMass});

    // Efficiency
    histos.add("h3dLambdaPtVsBDTScoreVsMCTruth", "h3dLambdaPtVsBDTScoreVsMCTruth", kTH3F, {axisPt, BDTScoreAxis, MCTruthAxis});
    histos.add("h3dAntiLambdaPtVsBDTScoreVsMCTruth", "h3dAntiLambdaPtVsBDTScoreVsMCTruth", kTH3F, {axisPt, BDTScoreAxis, MCTruthAxis});
    histos.add("h3dGammaPtVsBDTScoreVsMCTruth", "h3dGammaPtVsBDTScoreVsMCTruth", kTH3F, {axisPt, BDTScoreAxis, MCTruthAxis});
    histos.add("h3dK0ShortPtVsBDTScoreVsMCTruth", "h3dK0ShortPtVsBDTScoreVsMCTruth", kTH3F, {axisPt, BDTScoreAxis, MCTruthAxis});
  }
  
  // Process candidate and store properties in object
  template <typename TV0Object>
  void FillHistos(TV0Object const& v0)
  {
    bool fIsLambda = false;
    bool fIsGamma = false;
    //bool fIsAntiLambda = false;
    //bool fIsK0Short = false;

    // ML Output (Real data)
    histos.fill(HIST("hLambdaBDTScore"), v0.lambdaBDTScore());
    histos.fill(HIST("hGammaBDTScore"), v0.gammaBDTScore());
    //histos.fill(HIST("hAntiLambdaBDTScore"), v0.antilambdaBDTScore());
    //histos.fill(HIST("hK0ShortBDTScore"), v0.k0shortBDTScore());

    // Invariant Mass
    histos.fill(HIST("h2dLambdaMassVsBDTScore"), v0.mLambda(), v0.lambdaBDTScore());
    histos.fill(HIST("h2dGammaMassVsBDTScore"), v0.mGamma(), v0.gammaBDTScore());
    //histos.fill(HIST("h2dAntiLambdaMassVsBDTScore"), v0.mAntiLambda(), v0.antilambdaBDTScore());
    //histos.fill(HIST("h2dK0ShortMassVsBDTScore"), v0.mK0Short(), v0.k0shortBDTScore());
    
    // MC part
    if constexpr (requires { v0.pdgCode(); }){
      // ML Output (MC data)
      if (v0.pdgCode() == 3122){
        histos.fill(HIST("hMCTrueLambdaBDTScore"), v0.lambdaBDTScore());
        fIsLambda = true;
      } 
      if (v0.pdgCode() == 22){
        histos.fill(HIST("hMCTrueGammaBDTScore"), v0.gammaBDTScore());
        fIsGamma = true;
      }
      //if (v0.pdgCode() == -3122){
      // histos.fill(HIST("hMCTrueAntiLambdaBDTScore"), v0.antilambdaBDTScore());
      // fIsAntiLambda = false;
      //} 
      //if (v0.pdgCode() == 310){
      // histos.fill(HIST("hMCTrueK0ShortBDTScore"), v0.k0shortBDTScore());
      // fIsK0Short = false;
      //}

      // Efficiency
      histos.fill(HIST("h3dLambdaPtVsBDTScoreVsMCTruth"), v0.pt(), v0.lambdaBDTScore(), fIsLambda);
      histos.fill(HIST("h3dGammaPtVsBDTScoreVsMCTruth"), v0.pt(), v0.gammaBDTScore(), fIsGamma);
      //histos.fill(HIST("h3dAntiLambdaPtVsBDTScoreVsMCTruth"), v0.pt(), v0.antilambdaBDTScore(), fIsAntiLambda);
      //histos.fill(HIST("h3dK0ShortPtVsBDTScoreVsMCTruth"), v0.pt(), v0.k0shortBDTScore(), fIsK0Short);
    }
  }

  void processV0DerivedData(soa::Join<aod::StraCollisions, aod::StraCents>::iterator const& coll, V0DerivedDatas const& v0s)
  {
    histos.fill(HIST("hEventCentrality"), coll.centFT0C());
    for (auto& cand : v0s){ 
      FillHistos(cand); 
    }
  }
  void processV0StandardData(soa::Join<aod::Collisions, aod::CentFT0Cs>::iterator const& coll, V0OriginalDatas const& v0s)
  {
    histos.fill(HIST("hEventCentrality"), coll.centFT0C());
    for (auto& cand : v0s){ 
      FillHistos(cand); 
    }
  }
  PROCESS_SWITCH(MLSelectionQC, processV0DerivedData, "Process standard v0 data", false);
  PROCESS_SWITCH(MLSelectionQC, processV0StandardData, "Process derived v0 data", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<MLSelectionQC>(cfgc)};
}
