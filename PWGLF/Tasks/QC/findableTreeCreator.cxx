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
// Strangeness findable Tree Creator 
//
// --- deals with derived data that has been specifically
//     generated to do the findable exercise.
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    david.dobrigkeit.chinellato@cern.ch
//    gianni.shigeru.setoue.liveraro@cern.ch
//

#include <Math/Vector4D.h>
#include <cmath>
#include <array>
#include <cstdlib>

#include "Framework/runDataProcessing.h"
#include "Framework/RunningWorkflowInfo.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/ASoA.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessPIDTables.h"
#include "PWGLF/DataModel/LFStrangenessFindableTables.h"
#include "PWGLF/DataModel/LFParticleIdentification.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/McCollisionExtra.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "CCDB/BasicCCDBManager.h"
#include "PWGLF/Utils/v0SelectionBits.h"
#include "PWGLF/Utils/v0SelectionGroup.h"
#include "PWGLF/Utils/v0SelectionTools.h"

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

using recoStraCollisions = soa::Join<aod::StraCollisions, aod::StraEvSels, aod::StraCents, aod::StraRawCents_003, aod::StraCollLabels>;
using reconstructedV0s = soa::Join<aod::V0CoreMCLabels, aod::V0Cores, aod::V0FoundTags, aod::V0MCCollRefs, aod::V0CollRefs, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas>;
using reconstructedV0sNoMC = soa::Join<aod::V0Cores, aod::V0Extras>;

using dauTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackTPCPIDs>;

struct findableTreeCreator {
  Produces<aod::V0FindCands> v0FindCandidates;
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  // master PDG code selection
  Configurable<int> pdgCode{"pdgCode", 22, "PDG code to select"};
  Configurable<bool> skipITSonly{"skipITSonly", true, "skip reco V0s if an ITS-only (no TPC) prong present"};

  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for analysis"};
  ConfigurableAxis axisCentrality{"axisCentrality", {VARIABLE_WIDTH, 0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f}, "Centrality"};


  void init(InitContext const&)
  {
    // Event counting
    histos.add("hCentrality", "hCentrality", kTH1D, {axisCentrality});

    // Duplicate counting
    histos.add("hNRecoV0s", "hNRecoV0s", kTH1D, {{50, -0.5, 49.5f}});
    histos.add("hNRecoV0sWithTPC", "hNRecoV0sWithTPC", kTH1D, {{50, -0.5, 49.5f}});
    histos.add("hNRecoV0sWrongColl", "hNRecoV0sWrongColl", kTH1D, {{50, -0.5, 49.5f}});

    // For broad correctness check
    histos.add("hFoundVsTracksOK", "hFoundVsTracksOK", kTH2D, {{2, -0.5, 1.5f}, {2, -0.5, 1.5f}});

    // Global findable
    histos.add("h2dPtVsCentrality_Findable", "hPtVsCentrality_Findable", kTH2D, {axisCentrality, axisPt});

    // Acceptably (for svertexer) tracked
    histos.add("h2dPtVsCentrality_AcceptablyTracked", "h2dPtVsCentrality_AcceptablyTracked", kTH2D, {axisCentrality, axisPt});

    // Found in any capacity, including ITSonly
    histos.add("h2dPtVsCentrality_FoundAny", "h2dPtVsCentrality_FoundAny", kTH2D, {axisCentrality, axisPt});

    // Found with prongs with the TPC (typical analysis setting)
    histos.add("h2dPtVsCentrality_Found", "h2dPtVsCentrality_Found", kTH2D, {axisCentrality, axisPt});

    // Found in loop (may have duplicates, meant as cross-check too)
    histos.add("h2dPtVsCentrality_FoundInLoop", "h2dPtVsCentrality_FoundInLoop", kTH2D, {axisCentrality, axisPt});

  }

    // Helper struct to pass v0 information
  struct {
    int posITSCls;
    int negITSCls;
    uint32_t posITSClSize;
    uint32_t negITSClSize;
    uint8_t posTPCRows;
    uint8_t negTPCRows;
    float posTPCSigmaPi;
    float negTPCSigmaPi;
    float posTPCSigmaPr;
    float negTPCSigmaPr;
    float posTPCSigmaEl;
    float negTPCSigmaEl;
    float PxPos; 
    float PyPos; 
    float PzPos; 
    float PxNeg; 
    float PyNeg; 
    float PzNeg; 
    float X; 
    float Y; 
    float Z; 
    float pT;
    float posEta;
    float negEta;
    float v0Eta;
    float v0radius;
    float PA;
    float dcapostopv;
    float dcanegtopv;
    float dcaV0daughters;
    float dcav0topv;
    float PsiPair;
    float centrality;
  } Candidate;

  // Process candidate and store properties in object
  template <typename TV0Object, typename TCollision>
  void processCandidate(TCollision const& coll, TV0Object const& cand)
  {
    auto posTrackExtra = cand.template posTrackExtra_as<dauTracks>();
    auto negTrackExtra = cand.template negTrackExtra_as<dauTracks>();

    // Track quality
    Candidate.posITSCls = posTrackExtra.itsNCls();
    Candidate.negITSCls = negTrackExtra.itsNCls();
    Candidate.posITSClSize = posTrackExtra.itsClusterSizes();
    Candidate.negITSClSize = negTrackExtra.itsClusterSizes();
    Candidate.posTPCRows = posTrackExtra.tpcCrossedRows();
    Candidate.negTPCRows = negTrackExtra.tpcCrossedRows();

    // TPC PID
    Candidate.posTPCSigmaPi = posTrackExtra.tpcNSigmaPi();
    Candidate.negTPCSigmaPi = negTrackExtra.tpcNSigmaPi();
    Candidate.posTPCSigmaPr = posTrackExtra.tpcNSigmaPr();
    Candidate.negTPCSigmaPr = negTrackExtra.tpcNSigmaPr();
    Candidate.posTPCSigmaEl = posTrackExtra.tpcNSigmaEl();
    Candidate.negTPCSigmaEl = negTrackExtra.tpcNSigmaEl();

    // General
    Candidate.PxPos = cand.pxpos();
    Candidate.PyPos = cand.pypos();
    Candidate.PzPos = cand.pzpos();
    Candidate.PxNeg = cand.pxneg();
    Candidate.PyNeg = cand.pyneg();
    Candidate.PzNeg = cand.pzneg();
    Candidate.X = cand.x();
    Candidate.Y = cand.y();
    Candidate.Z = cand.z();
    Candidate.pT = cand.pt();
    Candidate.posEta = cand.positiveeta();
    Candidate.negEta = cand.negativeeta();
    Candidate.v0Eta = cand.eta();
    
    // Topological
    Candidate.v0radius = cand.v0radius();
    Candidate.PA = TMath::ACos(cand.v0cosPA());
    Candidate.dcapostopv = cand.dcapostopv();
    Candidate.dcanegtopv = cand.dcanegtopv();
    Candidate.dcaV0daughters = cand.dcaV0daughters();
    Candidate.dcav0topv = cand.dcav0topv();
    Candidate.PsiPair = cand.psipair();

    // Debug/Aditional
    Candidate.centrality = coll.centFT0C();

  }

  void processEvents(
    recoStraCollisions::iterator const& collision // reco collisions for collision counting
  )
  {
    histos.fill(HIST("hCentrality"), collision.centFT0C());
  }

  void processV0s(
    aod::V0MCCores::iterator const& v0,               // non-duplicated MC V0 decays
    soa::SmallGroups<reconstructedV0s> const& recv0s, // reconstructed versions of the v0
    recoStraCollisions const&,                        // reco collisions for de-reference
    aod::StraMCCollisions const&,                     // MC collisions for de-reference
    dauTracks const&                                  // daughter track extras
  )
  {
    int pdgCodePositive = 211;
    int pdgCodeNegative = -211;
    if (pdgCode == 3122)
      pdgCodePositive = 2212;
    if (pdgCode == -3122)
      pdgCodePositive = -2212;
    if (pdgCode == 22) {
      pdgCodePositive = -11;
      pdgCodeNegative = +11;
    }

    if (v0.pdgCode() != pdgCode || v0.pdgCodePositive() != pdgCodePositive || v0.pdgCodeNegative() != pdgCodeNegative)
      return;
    if (!v0.isPhysicalPrimary())
      return;

    float rapidity = 2.0;
    if (pdgCode == 310)
      rapidity = RecoDecay::y(std::array{v0.pxPosMC() + v0.pxNegMC(), v0.pyPosMC() + v0.pyNegMC(), v0.pzPosMC() + v0.pzNegMC()}, o2::constants::physics::MassKaonNeutral);
    if (pdgCode == 22)
      rapidity = RecoDecay::y(std::array{v0.pxPosMC() + v0.pxNegMC(), v0.pyPosMC() + v0.pyNegMC(), v0.pzPosMC() + v0.pzNegMC()}, o2::constants::physics::MassPhoton);
    if (pdgCode == 3122 || pdgCode == -3122)
      rapidity = RecoDecay::y(std::array{v0.pxPosMC() + v0.pxNegMC(), v0.pyPosMC() + v0.pyNegMC(), v0.pzPosMC() + v0.pzNegMC()}, o2::constants::physics::MassLambda0);

    if (std::abs(rapidity) > 0.5f)
      return;

    double ptmc = std::hypot(v0.pxPosMC() + v0.pxNegMC(), v0.pyPosMC() + v0.pyNegMC(), v0.pzPosMC() + v0.pzNegMC());

    // step 1: count number of times this candidate was actually reconstructed
    histos.fill(HIST("hNRecoV0s"), recv0s.size());
    bool hasWrongCollision = false;
    float centrality = 100.5f;
    bool hasBeenAcceptablyTracked = false;
    bool hasBeenFound = false;
    int nCandidatesWithTPC = 0;

    // MC flags
    int PDGCodeMother = -1;
    int V0ID = v0.particleIdMC();

    if constexpr (requires { v0.pdgCodeMother(); }) {
      PDGCodeMother = v0.pdgCodeMother();
    }

    for (auto& recv0 : recv0s) {
      if (recv0.v0Type() != 1)
        continue; // skip anything other than a standard V0

      // de-reference daughter track extras
      auto pTrack = recv0.posTrackExtra_as<dauTracks>();
      auto nTrack = recv0.negTrackExtra_as<dauTracks>();

      // skip ITS-only for simplicity
      if (skipITSonly) {
        if (!pTrack.hasTPC() || !nTrack.hasTPC())
          continue;
      }

      // define properties for this V0
      bool pTrackOK = false, nTrackOK = false; // tracks are acceptably tracked

      auto coll = recv0.straCollision_as<recoStraCollisions>();
      int mcCollID_fromCollision = coll.straMCCollisionId();
      int mcCollID_fromV0 = recv0.straMCCollisionId();
      if (mcCollID_fromCollision != mcCollID_fromV0) {
        hasWrongCollision = true;
      } else {
        // if this is a correctly collision-associated V0, take centrality from here
        // N.B.: this could still be an issue if collision <-> mc collision is imperfect
        centrality = coll.centFT0C();
      }

      if (
        (pTrack.hasTPC() && pTrack.hasITS()) ||     // full global track
        (pTrack.hasTPC() && pTrack.hasTOF()) ||     // TPC + TOF is accepted
        (pTrack.hasTPC() && pTrack.hasTRD()) ||     // TPC + TRD is accepted
        (!pTrack.hasTPC() && pTrack.itsNCls() >= 6) // long ITS-only
      ) {
        pTrackOK = true; // for this V0 only
      }
      if (
        (nTrack.hasTPC() && nTrack.hasITS()) ||
        (nTrack.hasTPC() && nTrack.hasTOF()) || // TPC + TOF is accepted
        (nTrack.hasTPC() && nTrack.hasTRD()) || // TPC + TRD is accepted
        (!nTrack.hasTPC() && nTrack.itsNCls() >= 6)) {
        nTrackOK = true; // for this V0 only
      }

      // determine if this V0 would go to analysis or not
      if (recv0.isFound() && pTrackOK && nTrackOK) { 
        // Found 
        nCandidatesWithTPC++;
        hasBeenFound = true;
        histos.fill(HIST("h2dPtVsCentrality_FoundInLoop"), centrality, ptmc);

        processCandidate(coll, recv0);
        // Filling TTree for findable offline exercise
        v0FindCandidates(Candidate.posITSCls, Candidate.negITSCls, Candidate.posITSClSize, Candidate.negITSClSize, Candidate.posTPCRows, Candidate.negTPCRows,
                      Candidate.posTPCSigmaPi, Candidate.negTPCSigmaPi, Candidate.posTPCSigmaPr, Candidate.negTPCSigmaPr,
                      Candidate.posTPCSigmaEl, Candidate.negTPCSigmaEl, 
                      Candidate.PxPos, Candidate.PyPos, Candidate.PzPos, Candidate.PxNeg, Candidate.PyNeg, Candidate.PzNeg,
                      Candidate.X, Candidate.Y, Candidate.Z, Candidate.pT,
                      Candidate.posEta, Candidate.negEta, Candidate.v0Eta,
                      Candidate.v0radius, Candidate.PA, Candidate.dcapostopv, Candidate.dcanegtopv, Candidate.dcaV0daughters, Candidate.dcav0topv, Candidate.PsiPair,
                      Candidate.centrality, PDGCodeMother, V0ID, hasBeenFound);

      }
      if (pTrackOK && nTrackOK && !hasBeenFound){
        // acceptably tracked
        hasBeenAcceptablyTracked = true;

        processCandidate(coll, recv0);
        // Filling TTree for findable offline exercise
        v0FindCandidates(Candidate.posITSCls, Candidate.negITSCls, Candidate.posITSClSize, Candidate.negITSClSize, Candidate.posTPCRows, Candidate.negTPCRows,
                      Candidate.posTPCSigmaPi, Candidate.negTPCSigmaPi, Candidate.posTPCSigmaPr, Candidate.negTPCSigmaPr,
                      Candidate.posTPCSigmaEl, Candidate.negTPCSigmaEl, 
                      Candidate.PxPos, Candidate.PyPos, Candidate.PzPos, Candidate.PxNeg, Candidate.PyNeg, Candidate.PzNeg,
                      Candidate.X, Candidate.Y, Candidate.Z, Candidate.pT,
                      Candidate.posEta, Candidate.negEta, Candidate.v0Eta,
                      Candidate.v0radius, Candidate.PA, Candidate.dcapostopv, Candidate.dcanegtopv, Candidate.dcaV0daughters, Candidate.dcav0topv, Candidate.PsiPair,
                      Candidate.centrality, PDGCodeMother, V0ID, hasBeenFound);
      }

    }
  
    // Major check 1: Findable versus found in some capacity
    histos.fill(HIST("h2dPtVsCentrality_Findable"), centrality, ptmc);
    if (hasBeenAcceptablyTracked) {
      histos.fill(HIST("h2dPtVsCentrality_AcceptablyTracked"), centrality, ptmc);
    }
    if (hasBeenFound) {
      histos.fill(HIST("h2dPtVsCentrality_Found"), centrality, ptmc);
    }
    if (hasWrongCollision) {
      histos.fill(HIST("hNRecoV0sWrongColl"), recv0s.size());
    }
  }

  PROCESS_SWITCH(findableTreeCreator, processEvents, "process collision counters", true);
  PROCESS_SWITCH(findableTreeCreator, processV0s, "process V0s", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<findableTreeCreator>(cfgc)};
}
