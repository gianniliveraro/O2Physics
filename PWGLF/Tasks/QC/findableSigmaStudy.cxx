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
// Strangeness sigma0 findable study
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
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessPIDTables.h"
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
using reconstructedV0s = soa::Join<aod::V0CoreMCLabels, aod::V0Cores, aod::V0FoundTags, aod::V0MCCollRefs, aod::V0CollRefs, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas, aod::V0MCMothers>;
using reconstructedV0sNoMC = soa::Join<aod::V0Cores, aod::V0Extras>;

using dauTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackTPCPIDs>;

struct findableSigmaStudy {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  Configurable<bool> skipITSonly{"skipITSonly", true, "skip reco V0s if an ITS-only (no TPC) prong present"};

  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for analysis"};
  ConfigurableAxis axisCentrality{"axisCentrality", {VARIABLE_WIDTH, 0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f}, "Centrality"};

  // Invariant Mass
  ConfigurableAxis axisSigmaMass{"axisSigmaMass", {200, 1.16f, 1.23f}, "M_{#Sigma^{0}} (GeV/c^{2})"};

  void init(InitContext const&)
  {

    // Event counting
    histos.add("hCentrality", "hCentrality", kTH1D, {axisCentrality});

    histos.add("h2dNRecoV0s", "h2dNRecoV0s", kTH2D, {{50, -0.5, 49.5f}, {50, -0.5, 49.5f}});

    // Global findable
    histos.add("h2dPtVsCentrality_Findable", "hPtVsCentrality_Findable", kTH2D, {axisCentrality, axisPt});

    // Acceptably (for svertexer) tracked
    histos.add("h2dPtVsCentrality_AcceptablyTracked", "h2dPtVsCentrality_AcceptablyTracked", kTH2D, {axisCentrality, axisPt});

    // Found with prongs with the TPC (typical analysis setting)
    histos.add("h2dPtVsCentrality_Found", "h2dPtVsCentrality_Found", kTH2D, {axisCentrality, axisPt});

    // Mass Histogram for sanity test
    histos.add("hSigmaMass", "hSigmaMass", kTH1D, {axisSigmaMass});

  }

  // Process candidate
  template <typename TV0Object>
  std::pair<bool, bool> processParticle(TV0Object const& recov0){

    bool hasBeenFound = false;
    bool hasBeenAcceptablyTracked = false;
    // de-reference daughter track extras
    auto pTrack = recov0.template posTrackExtra_as<dauTracks>();
    auto nTrack = recov0.template negTrackExtra_as<dauTracks>();

    // skip ITS-only for simplicity
    if (skipITSonly) {
    if (!pTrack.hasTPC() || !nTrack.hasTPC())
        return {false, false};
    }

    // define properties for this V0
    bool pTrackOK = false, nTrackOK = false; // tracks are acceptably tracked

    if ((pTrack.hasTPC() && pTrack.hasITS()) || (pTrack.hasTPC() && pTrack.hasTOF()) || 
        (pTrack.hasTPC() && pTrack.hasTRD()) || (!pTrack.hasTPC() && pTrack.itsNCls() >= 6)) {
        pTrackOK = true;
    }
    if ((nTrack.hasTPC() && nTrack.hasITS()) || (nTrack.hasTPC() && nTrack.hasTOF()) || 
        (nTrack.hasTPC() && nTrack.hasTRD()) || (!nTrack.hasTPC() && nTrack.itsNCls() >= 6)) {
        nTrackOK = true;
    }

    if (pTrackOK && nTrackOK)
        hasBeenAcceptablyTracked = true;

    // cross-check correctness of new getter
    if (pTrack.hasITSTracker() && (pTrack.hasITS() && pTrack.itsChi2PerNcl() < -1e-3)) {
    LOGF(fatal, "Positive track: inconsistent outcome of ITS tracker getter and explicit check!");
    }
    if (nTrack.hasITSTracker() && (pTrack.hasITS() && nTrack.itsChi2PerNcl() < -1e-3)) {
    LOGF(fatal, "Negative track: inconsistent outcome of ITS tracker getter and explicit check!");
    }
    if (pTrack.hasITSAfterburner() && (pTrack.hasITS() && pTrack.itsChi2PerNcl() > -1e-3)) {
    LOGF(fatal, "Positive track: inconsistent outcome of ITS tracker getter and explicit check!");
    }
    if (nTrack.hasITSAfterburner() && (pTrack.hasITS() && nTrack.itsChi2PerNcl() > -1e-3)) {
    LOGF(fatal, "Negative track: inconsistent outcome of ITS tracker getter and explicit check!");
    }

    // determine if this V0 would go to analysis or not
    if (recov0.isFound() && pTrackOK && nTrackOK) { 
        hasBeenFound = true;
    }

    return {hasBeenFound, hasBeenAcceptablyTracked};
  }

  void processEvents(
    recoStraCollisions::iterator const& collision // reco collisions for collision counting
  )
  {
    histos.fill(HIST("hCentrality"), collision.centFT0C());
  }


  void processV0s(
    recoStraCollisions const& stracoll,      // reco collisions for de-reference
    aod::V0MCCores const& v0,               // non-duplicated MC V0 decays
    soa::SmallGroups<reconstructedV0s> const& recv0s, // reconstructed versions of the v0         
    aod::StraMCCollisions const&, // MC collisions for de-reference
    dauTracks const&                                  // daughter track extras
 )
 {
    float centrality = 85.0f; //stracoll.centFT0C();
    for (auto& gamma : v0) { // selecting photons from Sigma0
        bool isFindableGamma = false;
        bool hasBeenFoundGamma = false;
        bool hasBeenAcceptablyTrackedGamma = false;
        int GammaMotherID = -1;
        
        if (gamma.pdgCode() != 22 || gamma.pdgCodeMother() != 3212) {
            continue;
        }
        if (!gamma.isPhysicalPrimary())
            continue;

        float rapidityGamma = RecoDecay::y(std::array{gamma.pxPosMC() + gamma.pxNegMC(), gamma.pyPosMC() + gamma.pyNegMC(), gamma.pzPosMC() + gamma.pzNegMC()}, o2::constants::physics::MassPhoton);
        if (std::abs(rapidityGamma) > 0.5f)
            continue;

        //double ptmcGamma = std::hypot(gamma.pxPosMC() + gamma.pxNegMC(), gamma.pyPosMC() + gamma.pyNegMC(), gamma.pzPosMC() + gamma.pzNegMC());
        auto NRecoGamma = recv0s.size();
        for (auto& recgamma : recv0s) {
            if (recgamma.v0Type() != 1)
                continue; // skip anything other than a standard V0
            
            auto result = processParticle(recgamma);
            hasBeenFoundGamma = result.first;
            hasBeenAcceptablyTrackedGamma = result.second;
            GammaMotherID = recgamma.motherMCPartId();
        }
        //LOGF(info, "You have just found a gamma from sigma0!");
        isFindableGamma = true;
        // --------------------------- Lambda specific
        for (auto& lambda : v0) { // selecting lambdas from Sigma0
            bool isFindableLambda = false;
            bool hasBeenFoundLambda = false;
            bool hasBeenAcceptablyTrackedLambda = false;
            int LambdaMotherID = 0;

            if (lambda.pdgCode() != 3122 || lambda.pdgCodeMother() != 3212) {
                continue;
            }
            
            if (!lambda.isPhysicalPrimary())
                continue;

            float rapidityLambda = RecoDecay::y(std::array{lambda.pxPosMC() + lambda.pxNegMC(), lambda.pyPosMC() + lambda.pyNegMC(), lambda.pzPosMC() + lambda.pzNegMC()}, o2::constants::physics::MassLambda0);
            if (std::abs(rapidityLambda) > 0.5f)
                continue;

            //double ptmcLambda = std::hypot(lambda.pxPosMC() + lambda.pxNegMC(), lambda.pyPosMC() + lambda.pyNegMC(), lambda.pzPosMC() + lambda.pzNegMC());
            
            auto NRecoLambda = recv0s.size();
            for (auto& reclambda : recv0s) {
                if (reclambda.v0Type() != 1)
                    continue; // skip anything other than a standard V0

                LambdaMotherID = reclambda.motherMCPartId();
                
                auto result2 = processParticle(reclambda);
                hasBeenFoundLambda = result2.first;
                hasBeenAcceptablyTrackedLambda = result2.second;
            }

            histos.fill(HIST("h2dNRecoV0s"), NRecoGamma, NRecoLambda);

            if (GammaMotherID != LambdaMotherID) { // lambda and gamma from the same sigma0
               continue;
            }
            
            isFindableLambda = true;

            double ptmcSigma = std::hypot(gamma.pxPosMC() + gamma.pxNegMC() + lambda.pxPosMC() + lambda.pxNegMC(), 
                                     gamma.pyPosMC() + gamma.pyNegMC() + lambda.pyPosMC() + lambda.pyNegMC(), 
                                     gamma.pzPosMC() + gamma.pzNegMC() + lambda.pzPosMC() + lambda.pzNegMC());
            
            std::array<float, 3> pVecPhotons{gamma.pxPosMC() + gamma.pxNegMC(), gamma.pyPosMC() + gamma.pyNegMC(), gamma.pzPosMC() + gamma.pzNegMC()};
            std::array<float, 3> pVecLambda{lambda.pxPosMC() + lambda.pxNegMC(), lambda.pyPosMC() + lambda.pyNegMC(), lambda.pzPosMC() + lambda.pzNegMC()};
            auto arrMom = std::array{pVecPhotons, pVecLambda};
            float sigmaMass = RecoDecay::m(arrMom, std::array{o2::constants::physics::MassPhoton, o2::constants::physics::MassLambda0});

            // Fill final histograms 

            LOGF(info, "Okay! We have a sigma0!");
            histos.fill(HIST("hSigmaMass"), sigmaMass);

            if (isFindableGamma && isFindableLambda) {
                histos.fill(HIST("h2dPtVsCentrality_Findable"), centrality, ptmcSigma);
            } 
            if (hasBeenFoundGamma && hasBeenFoundLambda) {
                //LOGF(info, "You have just found a sigma0! Congratulations!");
                histos.fill(HIST("h2dPtVsCentrality_Found"), centrality, ptmcSigma);
            } 
            if (hasBeenAcceptablyTrackedGamma && hasBeenAcceptablyTrackedLambda) {
                histos.fill(HIST("h2dPtVsCentrality_AcceptablyTracked"), centrality, ptmcSigma);
            }
        }    
    }
  }
  
 PROCESS_SWITCH(findableSigmaStudy, processEvents, "process collision counters", true);
 PROCESS_SWITCH(findableSigmaStudy, processV0s, "process V0s", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<findableSigmaStudy>(cfgc)};
}
