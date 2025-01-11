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
// This code does basic QA of strangeness derived data
//  *+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*
//  Strange Derived Generation QA
//  *+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    gianni.shigeru.setoue.liveraro@cern.ch
//

#include <Math/Vector4D.h>
#include <cmath>
#include <array>
#include <cstdlib>

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "CommonConstants/PhysicsConstants.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace std;
using std::array;
using dauTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackTPCPIDs>;
using StrCollisionsDatas = soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels>; 
using V0DerivedDatas = soa::Join<aod::V0Cores, aod::V0CollRefs, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas>;
using V0DerivedMCDatas = soa::Join<aod::V0Cores, aod::V0CollRefs, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas, aod::V0MCDatas>;
using CascDerivedMCDatas = soa::Join<aod::CascCollRefs, aod::CascCores, aod::CascExtras, aod::CascTOFPIDs, aod::CascTOFNSigmas, aod::CascBBs, aod::CascCoreMCLabels>;
using CascDerivedDatas = soa::Join<aod::CascCollRefs, aod::CascCores, aod::CascExtras, aod::CascTOFPIDs, aod::CascTOFNSigmas, aod::CascBBs>

struct strderivedGenQA {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  ConfigurableAxis axisNCollisions{"axisNCollisions", {50000, -0.5f, 49999.5f}, "collisions"};
  ConfigurableAxis axisNV0s{"axisNV0s", {50000, -0.5f, 49999.5f}, "V0s"};
  Configurable<bool> verbose{"verbose", false, "do more printouts"};

  // pack track quality but separte also afterburner
  // dynamic range: 0-31
  enum selection : int { hasTPC = 0,
                         hasITSTracker,
                         hasITSAfterburner,
                         hasTRD,
                         hasTOF };

  void init(InitContext const&)
  {

    histos.add("h3dTrackPropertiesVspT", "h3dTrackPropertiesVspT", kTH3D, {{32, -0.5, 31.5f}, {32, -0.5, 31.5f}, axisPt});
    histos.add("h3dTrackPropertiesVsCentrality", "h3dTrackPropertiesVsCentrality", kTH3D, {{32, -0.5, 31.5f}, {32, -0.5, 31.5f}, axisCentrality});


    // Add histogram to the list
    histos.add("hTrackCode", "hTrackCode", kTH1F, {{32, -0.5, 31.5f}});

    // Set bin labels for all combinations
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(1, "None");                            // Code 0
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(2, "TPC");                             // Code 1
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(3, "ITSTracker");                      // Code 2
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(4, "ITSTracker + TPC");               // Code 3
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(5, "ITSAfterburner");                 // Code 4
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(6, "ITSAfterburner + TPC");           // Code 5
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(7, "ITSAfterburner + ITSTracker");    // Code 6
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(8, "ITSAfterburner + ITSTracker + TPC"); // Code 7
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(9, "TRD");                            // Code 8
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(10, "TRD + TPC");                     // Code 9
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(11, "TRD + ITSTracker");              // Code 10
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(12, "TRD + ITSTracker + TPC");        // Code 11
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(13, "TRD + ITSAfterburner");          // Code 12
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(14, "TRD + ITSAfterburner + TPC");    // Code 13
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(15, "TRD + ITSAfterburner + ITSTracker"); // Code 14
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(16, "TRD + ITSAfterburner + ITSTracker + TPC"); // Code 15
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(17, "TOF");                           // Code 16
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(18, "TOF + TPC");                    // Code 17
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(19, "TOF + ITSTracker");             // Code 18
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(20, "TOF + ITSTracker + TPC");       // Code 19
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(21, "TOF + ITSAfterburner");         // Code 20
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(22, "TOF + ITSAfterburner + TPC");   // Code 21
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(23, "TOF + ITSAfterburner + ITSTracker"); // Code 22
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(24, "TOF + ITSAfterburner + ITSTracker + TPC"); // Code 23
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(25, "TOF + TRD");                    // Code 24
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(26, "TOF + TRD + TPC");             // Code 25
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(27, "TOF + TRD + ITSTracker");      // Code 26
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(28, "TOF + TRD + ITSTracker + TPC"); // Code 27
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(29, "TOF + TRD + ITSAfterburner");  // Code 28
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(30, "TOF + TRD + ITSAfterburner + TPC"); // Code 29
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(31, "TOF + TRD + ITSAfterburner + ITSTracker"); // Code 30
    histos.get<TH1>(HIST("hTrackCode"))->GetXaxis()->SetBinLabel(32, "All");                         // Code 31

  }

  void processDerivedV0s(StrCollisionsDatas::iterator const& coll, V0DerivedDatas const& V0s, dauTracks const&)
  {
    // Event Level
    float centrality = coll.centFT0C();
    histos.fill(HIST("Event/hPosZ"), coll.posZ()); 
    histos.fill(HIST("Event/hSel8"), coll.sel8()); 
    histos.fill(HIST("Event/h2dMultFT0C"), centrality, coll.multFT0C()); 
    histos.fill(HIST("Event/h2dMultNTracksPVeta1"), centrality, coll.multNTracksPVeta1()); 
    histos.fill(HIST("Event/h2dMultPVTotalContributors"), centrality, coll.multPVTotalContributors()); 
    histos.fill(HIST("Event/h2dMultAllTracksTPCOnly"), centrality, coll.multAllTracksTPCOnly()); 
    histos.fill(HIST("Event/h2dMultAllTracksITSTPC"), centrality, coll.multAllTracksITSTPC()); 
    histos.fill(HIST("Event/h2dNumTracksInTimeRange"), centrality, coll.numTracksInTimeRange()); 
    histos.fill(HIST("Event/h2dNumV0sPerColl"), centrality, V0s.size()); 
    
    for (auto const& v0 : V0s) {

      // V0-Level
      float V0Y_Gamma = RecoDecay::y(std::array{v0.px(), v0.py(), v0.pz()}, o2::constants::physics::MassGamma);
      float V0Y_Lambda = RecoDecay::y(std::array{v0.px(), v0.py(), v0.pz()}, o2::constants::physics::MassLambda);
      float V0Y_K0Short = RecoDecay::y(std::array{v0.px(), v0.py(), v0.pz()}, o2::constants::physics::MassK0Short);

      float pT = v0.pt();
      histos.fill(HIST("V0/hpT"), pT); 
      histos.fill(HIST("V0/h2dArmenterosP"), v0.alpha(), v0.qtarm()); 
      histos.fill(HIST("V0/hRadius"), v0.v0radius()); 
      histos.fill(HIST("V0/hZ"), v0.z()); 
      histos.fill(HIST("V0/hCosPA"), v0.v0cosPA()); 
      histos.fill(HIST("V0/hdcaDau"), v0.dcaV0daughters()); 
      histos.fill(HIST("V0/hdcaNegtopv"), v0.dcanegtopv()); 
      histos.fill(HIST("V0/hdcaPostopv"), v0.dcapostopv()); 
      histos.fill(HIST("V0/h2dEtaPhi"), v0.eta(), RecoDecay::phi(v0.px(), v0.py())); 
      histos.fill(HIST("V0/hYGamma"), V0Y_Gamma);
      histos.fill(HIST("V0/hYLambda"), V0Y_Lambda);
      histos.fill(HIST("V0/hYK0Short"), V0Y_K0Short);
      histos.fill(HIST("V0/hMassGamma"), v0.mGamma());
      histos.fill(HIST("V0/hMassLambda"), v0.mLambda());
      histos.fill(HIST("V0/hMassK0Short"), v0.mK0Short());
      histos.fill(HIST("V0/hV0Type"), v0.v0Type());
      histos.fill(HIST("V0/h2dV0Indices"), v0.straCollisionId(), v0.globalIndex()); // cross-check index correctness
    
      // Track-level
      auto posTrack = v0.template posTrackExtra_as<dauTracks>();
      auto negTrack = v0.template negTrackExtra_as<dauTracks>();

      uint8_t positiveTrackCode = ((uint8_t(posTrack.hasTPC()) << hasTPC) |
                                    (uint8_t(posTrack.hasITSTracker()) << hasITSTracker) |
                                    (uint8_t(posTrack.hasITSAfterburner()) << hasITSAfterburner) |
                                    (uint8_t(posTrack.hasTRD()) << hasTRD) |
                                    (uint8_t(posTrack.hasTOF()) << hasTOF));

      uint8_t negativeTrackCode = ((uint8_t(negTrack.hasTPC()) << hasTPC) |
                                    (uint8_t(negTrack.hasITSTracker()) << hasITSTracker) |
                                    (uint8_t(negTrack.hasITSAfterburner()) << hasITSAfterburner) |
                                    (uint8_t(negTrack.hasTRD()) << hasTRD) |
                                    (uint8_t(negTrack.hasTOF()) << hasTOF));

      histos.fill(HIST("V0/Track/h3dITSNCls"), centrality, v0.positivept(), posTrack.itsNCls());
      histos.fill(HIST("V0/Track/h3dITSNCls"), centrality, -1*v0.negativept(), negTrack.itsNCls());
      histos.fill(HIST("V0/Track/h3dITSChi2PerNcl"), centrality, v0.positivept(), posTrack.itsChi2PerNcl());
      histos.fill(HIST("V0/Track/h3dITSChi2PerNcl"), centrality, -1*v0.negativept(), negTrack.itsChi2PerNcl());
      histos.fill(HIST("V0/Track/h3dTPCCrossedRows"), centrality, v0.positivept(), posTrack.tpcCrossedRows());
      histos.fill(HIST("V0/Track/h3dTPCCrossedRows"), centrality, -1*v0.negativept(), negTrack.tpcCrossedRows());
      histos.fill(HIST("V0/Track/h3dTrackPropertiesVspT"), positiveTrackCode, negativeTrackCode, pT); // tracking complete info
      histos.fill(HIST("V0/Track/h2dPosTrackProperties"), positiveTrackCode, v0.positivept()); // pos track info
      histos.fill(HIST("V0/Track/h2dNegTrackProperties"), negativeTrackCode, v0.negativept()); // neg track info

      // PID (TPC)
      histos.fill(HIST("V0/PID/h3dTPCNSigmaEl"), centrality, v0.positivept(), posTrack.tpcNSigmaEl());
      histos.fill(HIST("V0/PID/h3dTPCNSigmaEl"), centrality, -1*v0.negativept(), negTrack.tpcNSigmaEl());
      histos.fill(HIST("V0/PID/h3dTPCNSigmaPr"), centrality, v0.positivept(), posTrack.tpcNSigmaPr());
      histos.fill(HIST("V0/PID/h3dTPCNSigmaPr"), centrality, -1*v0.negativept(), negTrack.tpcNSigmaPr());
      histos.fill(HIST("V0/PID/h3dTPCNSigmaPi"), centrality, v0.positivept(), posTrack.tpcNSigmaPi());
      histos.fill(HIST("V0/PID/h3dTPCNSigmaPi"), centrality, -1*v0.negativept(), negTrack.tpcNSigmaPi());
      histos.fill(HIST("V0/PID/h3dTPCSignal"), centrality, v0.positivept(), posTrack.tpcSignal());
      histos.fill(HIST("V0/PID/h3dTPCSignal"), centrality, -1*v0.negativept(), negTrack.tpcSignal());

      // PID (TOF)
      histos.fill(HIST("V0/PID/h3dTOFNSigmaLaPr"), centrality, pT, v0.tofNSigmaLaPr());
      histos.fill(HIST("V0/PID/h3dTOFNSigmaLaPi"), centrality, pT, v0.tofNSigmaLaPi());
      histos.fill(HIST("V0/PID/h3dTOFDeltaTLaPr"), centrality, pT, v0.posTOFDeltaTLaPr());
      histos.fill(HIST("V0/PID/h3dTOFDeltaTLaPi"), centrality, pT, v0.negTOFDeltaTLaPi());
      histos.fill(HIST("V0/PID/h3dTOFNSigmaALaPr"), centrality, pT, v0.tofNSigmaALaPr());
      histos.fill(HIST("V0/PID/h3dTOFNSigmaALaPi"), centrality, pT, v0.tofNSigmaALaPi());
      histos.fill(HIST("V0/PID/h3dTOFDeltaTALaPr"), centrality, pT, v0.posTOFDeltaTALaPr());
      histos.fill(HIST("V0/PID/h3dTOFDeltaTALaPi"), centrality, pT, v0.negTOFDeltaTALaPi());
      histos.fill(HIST("V0/PID/h3dTOFNSigmaK0PiPlus"), centrality, pT, v0.tofNSigmaK0PiPlus());
      histos.fill(HIST("V0/PID/h3dTOFNSigmaK0PiMinus"), centrality, pT, v0.tofNSigmaK0PiMinus());
      
      // PID TPC + TOF
      histos.fill(HIST("V0/PID/h3dTPCVsTOFNSigmaLaPr"), posTrack.tpcNSigmaPr(), v0.tofNSigmaLaPr(), v0.positivept());
      histos.fill(HIST("V0/PID/h3dTPCVsTOFNSigmaLaPi"), negTrack.tpcNSigmaPi(), v0.tofNSigmaLaPi(), v0.negativept());
    }
    
  }

  void processMCDerivedV0s(StrCollisionsDatas::iterator const& coll, V0DerivedMCDatas const& V0s, dauTracks const&)
  {
    // Event Level
    float centrality = coll.centFT0C();

    for (auto const& v0 : V0s) {
      // General 
      histos.fill(HIST("MCV0/h2dPDGV0VsMother"), v0.pdgCode(), v0.pdgCodeMother());
      histos.fill(HIST("MCV0/h2dPDGV0VsPositive"), v0.pdgCode(), v0.pdgCodePositive());
      histos.fill(HIST("MCV0/h2dPDGV0VsNegative"), v0.pdgCode(), v0.pdgCodeNegative());
      histos.fill(HIST("MCV0/h2dPDGV0VsIsPhysicalPrimary"), v0.pdgCode(), v0.isPhysicalPrimary());

      // Track-level
      auto posTrack = v0.template posTrackExtra_as<dauTracks>();
      auto negTrack = v0.template negTrackExtra_as<dauTracks>();

      // Specific analysis by species:
      if (v0.pdgCode==22){ // IsGamma
        histos.fill(HIST("MCV0/h2dArmenterosP"), v0.alpha(), v0.qtarm());
        histos.fill(HIST("MCV0/Gamma/h3dpTResolution"), centrality, v0.pt(), v0.pt()-v0.ptMC());
        histos.fill(HIST("MCV0/Gamma/h3dMass"), centrality, v0.pt(), v0.mGamma()); 
        histos.fill(HIST("MCV0/Gamma/h2dTPCNSigmaEl"), v0.positivept(), posTrack.tpcNSigmaEl());
        histos.fill(HIST("MCV0/Gamma/h2dTPCNSigmaEl"), -1*v0.negativept(), negTrack.tpcNSigmaEl());
        histos.fill(HIST("MCV0/Gamma/h2dTPCSignal"), v0.positivept(), posTrack.tpcSignal());
        histos.fill(HIST("MCV0/Gamma/h2dTPCSignal"), -1*v0.negativept(), negTrack.tpcSignal());
      }
      if (v0.pdgCode==3122){ // IsLambda
        histos.fill(HIST("MCV0/h2dArmenterosP"), v0.alpha(), v0.qtarm());
        histos.fill(HIST("MCV0/Lambda/h3dpTResolution"), centrality, v0.pt(), v0.pt()-v0.ptMC()); 
        histos.fill(HIST("MCV0/Lambda/h3dMass"), centrality, v0.pt(), v0.mLambda());
        histos.fill(HIST("MCV0/Lambda/h2dTPCNSigmaPr"), v0.positivept(), posTrack.tpcNSigmaPr());
        histos.fill(HIST("MCV0/Lambda/h2dTPCNSigmaPi"), -1*v0.negativept(), negTrack.tpcNSigmaPi());
        histos.fill(HIST("MCV0/Lambda/h2dTPCSignal"), v0.positivept(), posTrack.tpcSignal());
        histos.fill(HIST("MCV0/Lambda/h2dTPCSignal"), -1*v0.negativept(), negTrack.tpcSignal());
      }
      if (v0.pdgCode==-3122){ // IsAntiLambda
        histos.fill(HIST("MCV0/h2dArmenterosP"), v0.alpha(), v0.qtarm());
        histos.fill(HIST("MCV0/AntiLambda/h3dpTResolution"), centrality, v0.pt(), v0.pt()-v0.ptMC());         
        histos.fill(HIST("MCV0/AntiLambda/h3dMass"), centrality, v0.pt(), v0.mAntiLambda());
        histos.fill(HIST("MCV0/AntiLambda/h2dTPCNSigmaPr"), -1*v0.negativept(), negTrack.tpcNSigmaPr());
        histos.fill(HIST("MCV0/AntiLambda/h2dTPCNSigmaPi"), v0.positivept(), posTrack.tpcNSigmaPi());
        histos.fill(HIST("MCV0/AntiLambda/h2dTPCSignal"), v0.positivept(), posTrack.tpcSignal());
        histos.fill(HIST("MCV0/AntiLambda/h2dTPCSignal"), -1*v0.negativept(), negTrack.tpcSignal());
      }
      if (v0.pdgCode==310){ // IsK0Short
        histos.fill(HIST("MCV0/h2dArmenterosP"), v0.alpha(), v0.qtarm());
        histos.fill(HIST("MCV0/K0Short/h3dpTResolution"), centrality, v0.pt(), v0.pt()-v0.ptMC()); 
        histos.fill(HIST("MCV0/K0Short/h3dMass"), centrality, v0.pt(), v0.mK0Short());
        histos.fill(HIST("MCV0/K0Short/h2dTPCNSigmaPi"), v0.positivept(), posTrack.tpcNSigmaPi());
        histos.fill(HIST("MCV0/K0Short/h2dTPCNSigmaPi"), -1*v0.negativept(), negTrack.tpcNSigmaPi());
        histos.fill(HIST("MCV0/K0Short/h2dTPCSignal"), v0.positivept(), posTrack.tpcSignal());
        histos.fill(HIST("MCV0/K0Short/h2dTPCSignal"), -1*v0.negativept(), negTrack.tpcSignal());
      }
    }
  }

  void processDerivedCascades(StrCollisionsDatas::iterator const& coll, CascDerivedDatas const& Cascades, dauTracks const&)
  {
    for (auto& casc : Cascades) {
      
      // Cascade level
      float pT = casc.pt();
      histos.fill(HIST("Casc/Sign"), casc.sign()); 
      histos.fill(HIST("Casc/hpT"), pT);       
      histos.fill(HIST("Casc/hV0Radius"), casc.v0radius()); 
      histos.fill(HIST("Casc/hCascRadius"), casc.cascradius()); 
      histos.fill(HIST("Casc/hV0CosPA"), casc.v0cosPA()); 
      histos.fill(HIST("Casc/hCascCosPA"), casc.casccosPA()); 
      histos.fill(HIST("Casc/hDCAPosToPV"), casc.dcapostopv()); 
      histos.fill(HIST("Casc/hDCANegToPV"), casc.dcanegtopv()); 
      histos.fill(HIST("Casc/hDCABachToPV"), casc.dcabachtopv()); 
      histos.fill(HIST("Casc/hDCAXYCascToPV"), casc.dcaXYCascToPV()); 
      histos.fill(HIST("Casc/hDCAZCascToPV"), casc.dcaZCascToPV()); 
      histos.fill(HIST("Casc/hDCAV0ToPV"), casc.dcav0topv()); 
      histos.fill(HIST("Casc/hDCAV0Dau"), casc.dcaV0daughters());
      histos.fill(HIST("Casc/hDCACascDau"), casc.dcacascdaughters());
      histos.fill(HIST("Casc/hLambdaMass"), casc.mLambda());      
      
      // Track level
      auto negTrack = casc.template negTrackExtra_as<dauTracks>();
      auto posTrack = casc.template posTrackExtra_as<dauTracks>();
      auto bachTrack = casc.template bachTrackExtra_as<dauTracks>();

      uint8_t positiveTrackCode = ((uint8_t(posTrack.hasTPC()) << hasTPC) |
                                    (uint8_t(posTrack.hasITSTracker()) << hasITSTracker) |
                                    (uint8_t(posTrack.hasITSAfterburner()) << hasITSAfterburner) |
                                    (uint8_t(posTrack.hasTRD()) << hasTRD) |
                                    (uint8_t(posTrack.hasTOF()) << hasTOF));

      uint8_t negativeTrackCode = ((uint8_t(negTrack.hasTPC()) << hasTPC) |
                                    (uint8_t(negTrack.hasITSTracker()) << hasITSTracker) |
                                    (uint8_t(negTrack.hasITSAfterburner()) << hasITSAfterburner) |
                                    (uint8_t(negTrack.hasTRD()) << hasTRD) |
                                    (uint8_t(negTrack.hasTOF()) << hasTOF));

      uint8_t bachTrackCode = ((uint8_t(bachTrack.hasTPC()) << hasTPC) |
                                    (uint8_t(bachTrack.hasITSTracker()) << hasITSTracker) |
                                    (uint8_t(bachTrack.hasITSAfterburner()) << hasITSAfterburner) |
                                    (uint8_t(bachTrack.hasTRD()) << hasTRD) |
                                    (uint8_t(bachTrack.hasTOF()) << hasTOF));

      histos.fill(HIST("Casc/Track/h3dTrackProperties"), positiveTrackCode, negativeTrackCode, bachTrackCode); // complete tracking info
      histos.fill(HIST("Casc/Track/h2dPosTrackProperties"), positiveTrackCode, casc.positivept()); // positive track info
      histos.fill(HIST("Casc/Track/h2dNegTrackProperties"), negativeTrackCode, casc.negativept()); // negative track info
      histos.fill(HIST("Casc/Track/h2dBachTrackProperties"), bachTrackCode, casc.bachelorpt()); // bach track info
      histos.fill(HIST("Casc/Track/h3dV0ITSChi2PerNcl"), centrality, casc.positivept(), posTrack.itsChi2PerNcl());
      histos.fill(HIST("Casc/Track/h3dV0ITSChi2PerNcl"), centrality, -1*casc.negativept(), negTrack.itsChi2PerNcl());
      histos.fill(HIST("Casc/Track/h3dV0TPCCrossedRows"), centrality, casc.positivept(), posTrack.tpcCrossedRows());
      histos.fill(HIST("Casc/Track/h3dV0TPCCrossedRows"), centrality, -1*casc.negativept(), negTrack.tpcCrossedRows());
      histos.fill(HIST("Casc/Track/h3dV0ITSNCls"), centrality, casc.positivept(), posTrack.itsNCls());
      histos.fill(HIST("Casc/Track/h3dV0ITSNCls"), centrality, -1*casc.negativept(), negTrack.itsNCls());

      // PID (TPC)
      histos.fill(HIST("Casc/PID/h3dV0TPCNSigmaPr"), centrality, casc.positivept(), posTrack.tpcNSigmaPr());
      histos.fill(HIST("Casc/PID/h3dV0TPCNSigmaPr"), centrality, -1*casc.negativept(), negTrack.tpcNSigmaPr());
      histos.fill(HIST("Casc/PID/h3dV0TPCNSigmaPi"), centrality, casc.positivept(), posTrack.tpcNSigmaPi());
      histos.fill(HIST("Casc/PID/h3dV0TPCNSigmaPi"), centrality, -1*casc.negativept(), negTrack.tpcNSigmaPi());
      histos.fill(HIST("Casc/PID/h3dV0TPCSignal"), centrality, casc.positivept(), posTrack.tpcSignal());
      histos.fill(HIST("Casc/PID/h3dV0TPCSignal"), centrality, -1*casc.negativept(), negTrack.tpcSignal());

      // PID (TOF)
      histos.fill(HIST("Casc/PID/h3dTOFNSigmaXiLaPi"), centrality, pT, casc.tofNSigmaXiLaPi()); //! meson track NSigma from pion <- lambda <- xi expectation
      histos.fill(HIST("Casc/PID/h3dTOFNSigmaXiLaPr"), centrality, pT, casc.tofNSigmaXiLaPr()); //! baryon track NSigma from proton <- lambda <- xi expectation
      histos.fill(HIST("Casc/PID/h3dTOFNSigmaXiPi"), centrality, pT, casc.tofNSigmaXiPi());     //! bachelor track NSigma from pion <- xi expectation
      histos.fill(HIST("Casc/PID/h3dTOFNSigmaOmLaPi"), centrality, pT, casc.tofNSigmaOmLaPi()); //! meson track NSigma from pion <- lambda <- om expectation     
      histos.fill(HIST("Casc/PID/h3dTOFNSigmaOmLaPr"), centrality, pT, casc.tofNSigmaOmLaPr()); //! baryon track NSigma from proton <- lambda <- om expectation
      histos.fill(HIST("Casc/PID/h3dTOFNSigmaOmKa"), centrality, pT, casc.tofNSigmaOmKa());     //! bachelor track NSigma from kaon <- om expectation

      // By particle species
      if (casc.sign()<0){
        histos.fill(HIST("Casc/hMassXiMinus"), casc.mXi());
        histos.fill(HIST("Casc/hMassOmegaMinus"), v0.mOmega());
        histos.fill(HIST("Casc/Track/h3dBachITSNCls"), centrality, -1*casc.bachelorpt(), bachTrack.itsNCls());
        histos.fill(HIST("Casc/Track/h3dBachITSChi2PerNcl"), centrality, -1*casc.bachelorpt(), bachTrack.itsChi2PerNcl());
        histos.fill(HIST("Casc/Track/h3dBachTPCCrossedRows"), centrality, -1*casc.bachelorpt(), bachTrack.tpcCrossedRows());
        histos.fill(HIST("Casc/PID/h3dBachTPCNSigmaPr"), centrality, -1*casc.bachelorpt(), bachTrack.tpcNSigmaPr());
        histos.fill(HIST("Casc/PID/h3dBachTPCNSigmaPi"), centrality, -1*casc.bachelorpt(), bachTrack.tpcNSigmaPi());
        histos.fill(HIST("Casc/PID/h3dBachTPCSignal"), centrality, -1*casc.bachelorpt(), bachTrack.tpcSignal());
      }
      else{
        histos.fill(HIST("Casc/hMassXiPlus"), casc.mXi());
        histos.fill(HIST("Casc/hMassOmegaPlus"), v0.mOmega());
        histos.fill(HIST("Casc/Track/h3dBachITSNCls"), centrality, casc.bachelorpt(), bachTrack.itsNCls());
        histos.fill(HIST("Casc/Track/h3dBachITSChi2PerNcl"), centrality, casc.bachelorpt(), bachTrack.itsChi2PerNcl());
        histos.fill(HIST("Casc/Track/h3dBachTPCCrossedRows"), centrality, casc.bachelorpt(), bachTrack.tpcCrossedRows());
        histos.fill(HIST("Casc/PID/h3dBachTPCNSigmaPr"), centrality, casc.bachelorpt(), bachTrack.tpcNSigmaPr());
        histos.fill(HIST("Casc/PID/h3dBachTPCNSigmaPi"), centrality, casc.bachelorpt(), bachTrack.tpcNSigmaPi());
        histos.fill(HIST("Casc/PID/h3dBachTPCSignal"), centrality, casc.bachelorpt(), bachTrack.tpcSignal());
      }
    }
  }

  void processMCDerivedCascades(StrCollisionsDatas::iterator const& coll, CascDerivedMCDatas const& Cascades, dauTracks const&)
  {
    for (auto& casc : Cascades) {
      // General 
      histos.fill(HIST("MCCasc/h2dPDGV0VsMother"), casc.pdgCode(), casc.pdgCodeMother());
      histos.fill(HIST("MCCasc/h2dPDGV0VsPositive"), casc.pdgCode(), casc.pdgCodePositive());
      histos.fill(HIST("MCCasc/h2dPDGV0VsNegative"), casc.pdgCode(), casc.pdgCodeNegative());
      histos.fill(HIST("MCCasc/h2dPDGV0VsBach"), casc.pdgCode(), casc.pdgCodeBachelor());
      histos.fill(HIST("MCCasc/h2dPDGV0VsIsPhysicalPrimary"), casc.pdgCode(), casc.isPhysicalPrimary());

      // Track level
      auto negTrack = casc.template negTrackExtra_as<dauTracks>();
      auto posTrack = casc.template posTrackExtra_as<dauTracks>();
      auto bachTrack = casc.template bachTrackExtra_as<dauTracks>();

      // Specific analysis by species:
      if (casc.pdgCode==3312){ // XiMinus
      }
      if (casc.pdgCode==-3312){ // XiPlus
      }
      if (casc.pdgCode==3334){ // OmegaMinus
      }
      if (casc.pdgCode==-3334){ // OmegaPlus
      }
    }

  }

  PROCESS_SWITCH(strderivedGenQA, processDerivedV0s, "Process derived data", true);
  PROCESS_SWITCH(strderivedGenQA, processMCDerivedV0s, "Process derived data", false);
  PROCESS_SWITCH(strderivedGenQA, processDerivedCascades, "Process derived data", true);
  PROCESS_SWITCH(strderivedGenQA, processMCDerivedCascades, "Process derived data", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<strderivedGenQA>(cfgc)};
}


